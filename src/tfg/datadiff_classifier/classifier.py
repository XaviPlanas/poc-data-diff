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

    __CHAT = False
    __ANTHROPIC = True
     #__MODEL = 'claude-sonnet-4-5'
    __MODEL = 'claude-haiku-4-5'
    #__MODEL = 'phi3'
    #__MODEL = 'llama3:8b'
    #__MODEL = 'qwen2.5:7b-instruct'
    __TEMPERATURE = 0
    __PROMPT = CLASSIFY_PROMPT_SEMANTIC_ROW_DIFF
    __SCHEMA = SCHEMA_CONTEXT_JSON
    __LLM_UMBRAL_INCERTIDUMBRE=0.7
    #__FORMAT = JSON_CONSTRAINED_OUTPUT
    
    def __init__(self, schema_context: str = None):
        
        self.schema_context = schema_context or self.__class__.__SCHEMA

        # self.client = client = Client(host='http://127.0.0.1:11434')
        #     #host='https://ollama.com',
        #     #headers={'Authorization': 'Bearer ' +  os.environ.get('OLLAMA_API_KEY')} ) 
        self.client = anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))  
    
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
                  
    def classify(self, row: DiffRow, prompt = None) -> DiffClassification :
        
        prompt = prompt or self.__PROMPT.format(
            source_a=row.source_a,
            source_b=row.source_b,
            schema_context=self.schema_context,
            row_a=json.dumps(row.row_a, ensure_ascii=False),
            row_b=json.dumps(row.row_b, ensure_ascii=False),
        )

        try:   # Llamamos a la API del LLM y procesamos la respuesta en variable data
            if  self.__CHAT :     
                response = self.client.chat(
                model   = self.__MODEL,
                options = {"temperature": self.__TEMPERATURE},
                messages = [
                        {
                            "role":    "system",
                            "content": SYSTEM_PROMPT,
                        },
                        {
                            "role":    "user",
                            "content": self.__PROMPT,
                        },
                    ],
                )
                raw = response["message"]["content"].strip()
            elif self.__ANTHROPIC :
                response = self.client.messages.create(
                    model=self.__MODEL,
                    max_tokens=1024,
                    temperature=0.7,
                    messages=[
                    {"role": "user", "content": prompt}
                        ]
                    )
    
                raw = response.content[0].text.strip() 
            else : 
                response = self.client.generate(
                    model=self.__MODEL,             
                    prompt= prompt,
                    options={"temperature": self.__TEMPERATURE},
                    format="json"
                )
                raw = response["response"].strip()          
    
            print(f"Raw response from {self.__MODEL}:\n{raw}\n")
            
            match = re.search(r"\{.*\}", raw, re.DOTALL) #raw vienen como markdown JSON
            data = json.loads(match.group())
            
            # Construimos el DiffClassification resultante  
            # a partir del JSON devuelto por el LLM
            result = DiffClassification(
                key=row.key,
                accion = DiffAction(data.get("accion", DiffAction.UPDATE.value)),
                categoria=data.get("categoria", DiffCategory.UNCERTAIN.value),
                confianza=float(data.get("confianza", 0.0)),
                columnas_afectadas=data.get("columnas_afectadas", []),
                explicacion=data.get("explicacion", ""),
                normalizacion_sugerida=data.get("normalizacion_sugerida"),
                row_a=row.row_a,
                row_b=row.row_b
            )

            # Aplicar umbral de incertidumbre
            if result.confianza < self.__LLM_UMBRAL_INCERTIDUMBRE:
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
                categoria= DiffCategory.SEMANTICALLY_EQUIVALENT,
                confianza=1,
                columnas_afectadas=['*'],
                explicacion=f"{accion} del Identificador {row.key}",
                normalizacion_sugerida=None,
                row_a=row.row_a,
                row_b=row.row_b
            )
        return None
    
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

    def classify_all(self, diff_row: DiffRow) -> DiffClassification:
        
        prompt = self.__PROMPT.format(
            source_a        = diff_row.source_a,
            source_b        = diff_row.source_b,
            #schema_context  = self.schema_context,
            row_a           = json.dumps(diff_row.row_a, ensure_ascii=False),
            row_b           = json.dumps(diff_row.row_b, ensure_ascii=False),            
        )
        print(f"Utilizando el PROMPT: {prompt}")
        
        if not self.__CHAT : 
            response = self.client.generate(
                model= self.__MODEL,
                prompt= self.__PROMPT,
                options={
                #    "num_ctx": 8192,
                    "temperature": self.__TEMPERATURE,
                #     "top_p": 0.8,
                #     "top_k": 40,
                #     "num_predict": 100
                },
                format = "json"
            )

            print(response["response"])
            print("Total (ms):", response['total_duration']/1e6)
            print("Prompt eval (ms):", response['prompt_eval_duration']/1e6)
            print("Generation (ms):", response['eval_duration']/1e6)
            #print(json.dumps(response.model_dump(), indent=2))

        else :
            response = self.client.chat(
                model   = self.__MODEL,
                options = {"temperature": self.__TEMPERATURE},
                messages = [
                    {
                        "role":    "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role":    "user",
                        "content": self.__PROMPT,
                    },
                ],
                #format = self.__FORMAT
            )
            print(response["message"]["content"])

    def classify_one_row(self, row: DiffRow):

        prompt_text = self.__PROMPT.format(
            source_a=row.source_a,
            source_b=row.source_b,
            row_a=row.row_a or '{"__missing__": true}',
            row_b=row.row_b or '{"__missing__": true}'
        ).strip()

        clasificacion = self.__normalizador(row) # Filtra UPD 
        if clasificacion is None:
            clasificacion = self.classify(prompt=prompt_text, row=row)
        else : 
            if clasificacion.columnas_afectadas != ['*'] :  # INS o DEL 
                #print(f"Key: {row.key}. Row: {row}")
                print(f"Key: {row.key}.")
                if clasificacion.accion == DiffAction.INSERT :
                    print(f"Clasificacion : {clasificacion}\n")
        

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