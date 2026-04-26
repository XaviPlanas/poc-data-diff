from typing import Iterator
import json
from urllib import response
import traceback
import re 
import os 

from ollama import Client
import anthropic 

from tfg.datadiff_classifier.models import DiffEvent, DiffRow, DiffClassification, DiffCategory, DiffAction, SegmentStructure
from tfg.datadiff_classifier.prompts import CLASSIFY_PROMPT_FROM_DICT_OPTIMIZED_2, PROMPT_ROW_BY_ROW_2, CONSTRAINED_OUTPUT_ROW_BY_ROW_2, \
        CLASSIFY_PROMPT_FROM_DICT_OPTIMIZED_QWEN2_5b, JSON_CONSTRAINED_OUTPUT, SCHEMA_CONTEXT_JSON,  \
        SYSTEM_PROMPT, CLASSIFY_PROMPT, PROMPT_PHI3, CLASSIFY_PROMPT_FROM_DICT_OPTIMIZED_1, CLASSIFY_PROMPT_SEMANTIC_ROW_DIFF
from tfg.titanic_poc.titanic_utils import Config

class DiffClassifier:

    def __init__(self, 
                 schema_context: str = None, 
                 llm_provider: str = 'anthropic', 
                 model: str = 'claude-haiku-4-5', 
                 temperature: float = 0.0, 
                 api_key: str = None,
                 prompt_template: str = None,
                 uncertainty_threshold: float = 0.7,
                 ollama_use_chat: bool = False):
        
        self.schema_context = schema_context or SCHEMA_CONTEXT_JSON
        self.llm_provider = llm_provider
        self.model = model
        self.temperature = temperature
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        self.prompt_template = prompt_template or CLASSIFY_PROMPT_SEMANTIC_ROW_DIFF
        self.uncertainty_threshold = uncertainty_threshold
        self.ollama_use_chat = ollama_use_chat
        
        # Initialize client based on provider
        if self.llm_provider == 'anthropic':
            self.client = anthropic.Anthropic(api_key=self.api_key)
        elif self.llm_provider == 'ollama':
            self.client = Client(host='http://127.0.0.1:11434')
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")  
    
    def parse_diff_results(self, metadata: SegmentStructure, diffs: Iterator[dict]) -> list[DiffRow]    :
        """ Convierte la salida de data-diff en una lista de DiffRow, que es el formato de entrada esperado por el clasificador.
            Cada DiffRow representa una fila divergente, con toda la información de ambas filas y sus fuentes.
        """
        total = 0
        left = {}
        right = {}
        diffrows = []
        for diff in diffs : 
            total+=1
            d = dict(zip(metadata.columnas, diff[1][1:])) # [1::] es necesario si antes se extrae all_columns - pk?
            if diff[0] == '+' :
                right[d[metadata.pk]]=d 
            elif diff[0] == '-' :
                left[d[metadata.pk]]=d 

        right_idx = set(right.keys())
        left_idx = set(left.keys())
        insert = right_idx - left_idx
        delete = left_idx  - right_idx
        update = right_idx & left_idx
        all_pk = right_idx | left_idx

        if Config.DEBUG : 
            print(60*'=')
            print(f"Insertados   (INS) [{len(insert)}] : {insert}"  )
            print(f"Eliminados   (DEL) [{len(delete)}] : {delete}"  )
            print(f"Actualizados (UPD) [{len(update)}] : {update}"  )
            print(60*'-')
            print(f"Cambios detectados por data-diff en total (2UPD + DEL + INS): {total}")
            print(60*'=')

        # Construimos la lista de DiffRows (interfaz para classifier.py)
        for pk in all_pk:
            diffrows.append(DiffRow(
                key = pk,
                row_a = right.get(pk),
                row_b = left.get(pk),
                source_a = 'mysql',
                source_b = 'postgresql'
            ))
        return diffrows
                  
    def __IAsearch(self, row: DiffRow, prompt = None) -> str:
        """ Función auxiliar para llamar a la API del LLM con un prompt dado y devolver la respuesta JSON. """
        
        prompt = prompt or self.prompt_template.format(
            source_a=row.source_a,
            source_b=row.source_b,
            schema_context=self.schema_context,
            row_a=json.dumps(row.row_a, ensure_ascii=False),
            row_b=json.dumps(row.row_b, ensure_ascii=False),
        )

        if self.llm_provider == 'anthropic':
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=self.temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            raw = response.content[0].text.strip()
        elif self.llm_provider == 'ollama':
            if self.ollama_use_chat:
                response = self.client.chat(
                    model=self.model,
                    options={"temperature": self.temperature},
                    messages=[
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                )
                raw = response["message"]["content"].strip()
            else:
                response = self.client.generate(
                    model=self.model,             
                    prompt=prompt,
                    options={"temperature": self.temperature},
                    format="json"
                )
                raw = response["response"].strip()
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

        print(f"Raw response from {self.model}:\n{raw}\n")
        
        match = re.search(r"\{.*\}", raw, re.DOTALL) #raw vienen como markdown JSON
        data = json.loads(match.group())
        return data
          
    def IAclassify(self, row: DiffRow, prompt = None) -> DiffClassification :
        
        #Obtenmos la clasificación del LLM en formato JSON
        data = self.__IAsearch(row, prompt)
        #print(f"Data parsed from LLM response:\n{data}\n")
        try:
            # Mapear strings a enums 
            try:
                categoria_str = data.get("category", "UNCERTAIN")
                print(f"Categoria: {categoria_str}")
                categoria = DiffCategory(categoria_str)
            except ValueError:
                categoria = DiffCategory.ERROR
            
            # Construimos el DiffClassification resultante  
            # a partir del JSON devuelto por el LLM
            result = DiffClassification(
                key=row.key,
                accion= DiffAction.UPDATE,
                categoria=categoria,
                confianza=float(data.get("confidence_of_category", 0.0)),
                columnas_afectadas=data.get("affected_columns", []),
                explicacion=data.get("explanation", ""),
                normalizacion_sugerida=data.get("normalization_suggested"),
                row_a=row.row_a,
                row_b=row.row_b
            )

            # Aplicar umbral de incertidumbre
            if result.confianza < self.uncertainty_threshold:
                result.categoria = DiffCategory.UNCERTAIN
            
            return result
        
        except Exception as e:
            print(f"Error procesando key {row}: {e}")
            print(traceback.format_exc())
            return DiffClassification(
                key=row.key,
                accion = DiffAction.UPDATE,
                categoria = DiffCategory.ERROR,
                confianza=0.0,
                columnas_afectadas=[],
                explicacion=f"Error en LLM: {str(e)}",
                normalizacion_sugerida=None,
                row_a=row.row_a,
                row_b=row.row_b
            )
        
    def __normalizador(self,row : DiffRow ) -> DiffClassification | None :
        " Normalizador de casos obvios, como filas completamente ausentes (INSERT/DELETE) "
        " o con diferencias triviales."

        if row.row_a is None or row.row_b is None :
            if row.row_a is None:
                accion = DiffAction.DELETE
            else:
                accion = DiffAction.INSERT

            return DiffClassification(
                key=row.key,
                accion=accion,
                categoria= DiffCategory.SEMANTICALLY_DIFFERENT,
                confianza=1,
                columnas_afectadas=['*'],
                explicacion=f"{accion} del Identificador {row.key}",
                normalizacion_sugerida=None,
                row_a=row.row_a,
                row_b=row.row_b
            )
        return None
    
    def show_statistics(self, classifications: list[DiffClassification]) :
        """ Función auxiliar para obtener estadísticas de las clasificaciones. """
        stats = {
            DiffCategory.SEMANTICALLY_EQUIVALENT: 0,
            DiffCategory.SEMANTICALLY_DIFFERENT: 0,
            DiffCategory.UNCERTAIN: 0,
            DiffCategory.ERROR: 0,
            Total : 0
        }
        for c in classifications:
            stats[c.categoria] += 1
            stats['Total'] += 1
        
        print (f"Total diferencias clasificadas: {stats['Total']}")
        print (f"Diferencias clasificadas como SEMANTICALLY_DIFFERENT: {stats[DiffCategory.SEMANTICALLY_DIFFERENT]}") 
        print (f"Diferencias clasificadas como SEMANTICALLY_EQUIVALENT: {stats[DiffCategory.SEMANTICALLY_EQUIVALENT]}")
        print (f"Diferencias clasificadas como UNCERTAIN: {stats[DiffCategory.UNCERTAIN]}")      
        print (f"Diferencias clasificadas como ERROR: {stats[DiffCategory.ERROR]}")      


    def diff_to_json(diff: DiffClassification) -> str:
        return json.dumps({
            "key": diff.key,
            "accion": diff.accion.value if diff.accion else None,
            "categoria": diff.categoria.value if diff.categoria else None,
            "confianza": diff.confianza,
            "columnas_afectadas": diff.columnas_afectadas,
            "explicacion": diff.explicacion,
            "normalizacion_sugerida": diff.normalizacion_sugerida,
            "row_a": diff.row_a,
            "row_b": diff.row_b
        }, ensure_ascii=False, indent=2)
    
    def diffdata_to_events(diff: DiffRow) -> list[DiffEvent]:
        """ Del modelo basado en entidades a atributos:
         - DiffData representa una diferencia a nivel de fila, con toda la información de ambas filas.
         - DiffEvent representa una diferencia a nivel de columna, con el valor específico de esa columna en ambas fuentes.
        En la función se desglosan las diferencias fila a fila en eventos columna a columna, para luego clasificarlos individualmente.
        """
        eventos = []

        if diff.registro_a is None:
            # El registro no existía en A: es una inserción
            for col, val in diff.registro_b.items():
                eventos.append(DiffEvent(
                    key=diff.key,
                    columna=col,
                    valor_a=None,
                    valor_b=val,
                    accion=DiffAction.INSERT
                ))

        elif diff.registro_b is None:
            # El registro no existe en B: ha sido eliminado
            for col, val in diff.registro_a.items():
                eventos.append(DiffEvent(
                    key=diff.key,
                    columna=col,
                    valor_a=val,
                    valor_b=None,
                    accion=DiffAction.DELETE
                ))

        else:
            # El registro existe en ambas fuentes: comparar columna a columna
            columnas = set(diff.registro_a.keys()) | set(diff.registro_b.keys())
            for col in columnas:
                val_a = diff.registro_a.get(col)
                val_b = diff.registro_b.get(col)
                if val_a != val_b:
                    eventos.append(DiffEvent(
                        key=diff.key,
                        columna=col,
                        valor_a=val_a,
                        valor_b=val_b,
                        accion=DiffAction.UPDATE
                    ))

        return eventos

    def classify_all(self, diff_row: DiffRow) -> list[DiffClassification]:
        
        prompt = self.prompt_template.format(
            source_a        = diff_row.source_a,
            source_b        = diff_row.source_b,
            #schema_context  = self.schema_context,
            row_a           = json.dumps(diff_row.row_a, ensure_ascii=False),
            row_b           = json.dumps(diff_row.row_b, ensure_ascii=False),            
        )
        print(f"Utilizando el PROMPT: {prompt}")
        
        if self.llm_provider == 'ollama' and not self.ollama_use_chat:
            response = self.client.generate(
                model= self.model,
                prompt= prompt,
                options={
                    "temperature": self.temperature,
                },
                format = "json"
            )

            print(response["response"])
            print("Total (ms):", response['total_duration']/1e6)
            print("Prompt eval (ms):", response['prompt_eval_duration']/1e6)
            print("Generation (ms):", response['eval_∫∫duration']/1e6)

        elif self.llm_provider == 'ollama' and self.ollama_use_chat:
            response = self.client.chat(
                model   = self.model,
                options = {"temperature": self.temperature},
                messages = [
                    {
                        "role":    "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role":    "user",
                        "content": prompt,
                    },
                ],
            )
            print(response["message"]["content"])
        elif self.llm_provider == 'anthropic':
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            print(response.content[0].text.strip())
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    def classify_one_row(self, row: DiffRow) -> DiffClassification:

        prompt_text = self.prompt_template.format(
            source_a=row.source_a,
            source_b=row.source_b,
            row_a=row.row_a or '{"__missing__": true}',
            row_b=row.row_b or '{"__missing__": true}'
        ).strip()

        clasificacion = self.__normalizador(row) # Filtra UPD 
        
        if clasificacion is None: # pasamos a la IA los casos no resueltos por el normalizador 
            clasificacion = self.IAclassify(prompt=prompt_text, row=row)
           
        return clasificacion

    def classify_row_by_row ( self, diffrows: list[DiffRow] ) -> list[DiffClassification] :
        """
        Clasifica la lista de diferencias llamando al LLM fila por fila
        """
        diffresults = []
    
        for diff in diffrows:
            result = self.classify_one_row(diff)
            diffresults.append(result)
            print(f"Resultado clasificación fila {diff.key}: {result}\n")
        return diffresults

    def classify_batch(
        self, 
        diff_rows: list[DiffRow],
        confidence_threshold: float = 0.85
    ) -> list[DiffClassification]:
        """
        Clasifica una lista de diferencias. Las clasificaciones
        por debajo del umbral de confianza se marcan como UNCERTAIN
        para revisión manual.
        """
        results = []
        for row in diff_rows:
            result = self.classify(row)
            if result.confianza < confidence_threshold:
                result.categoria = DiffCategory.UNCERTAIN
            results.append(result)
        return results