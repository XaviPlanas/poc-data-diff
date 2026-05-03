from typing import Iterator, List, Optional
from textwrap import dedent
import json
import re 
import os 
from tenacity import retry, stop_after_attempt, wait_exponential

from ollama import Client
import anthropic 

from tfg.datadiff_classifier.models import DiffEvent, DiffRow, DiffClassification, DiffCategory, DiffAction, SegmentStructure
from tfg.datadiff_classifier.prompts import ( SYSTEM_PROMPT, SYSTEM_PROMPT_V2, 
                                             SCHEMA_CONTEXT_TITANIC, 
                                             USER_PROMPT_V2,
                                             FEW_SHOT_EXAMPLES
)

import logging
from tfg.logging_config import setup_logging, timed
logger = logging.getLogger(__name__)

class DiffClassifier:

    def __init__(self, 
                 schema_context: str = None, 
                 llm_provider: str = 'anthropic', 
                 model: str = 'claude-haiku-4-5', 
                 temperature: float = 0.0, 
                 api_key: str = None,
                 prompt_template: str = None,
                 few_shot: bool = True,
                 uncertainty_threshold: float = 0.7,
                 max_retries: int = 3):
        
        
        self.schema_context = schema_context or SCHEMA_CONTEXT_TITANIC
        self.llm_provider = llm_provider
        self.model = model
        self.temperature = temperature
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        self.prompt_template = prompt_template or USER_PROMPT_V2 
        self.few_shot = few_shot
        self.uncertainty_threshold = uncertainty_threshold
        # Initialize client based on provider
        if self.llm_provider == 'anthropic':
            logger.debug("Inicializando cliente Anthropic")
            self.client = anthropic.Anthropic(api_key=self.api_key)
        
            self._system = SYSTEM_PROMPT_V2.format(
                schema_context        = self.schema_context,
                uncertainty_threshold = self.uncertainty_threshold,
            )
        elif self.llm_provider == 'ollama':
            self.client = Client(host='http://127.0.0.1:11434')
        else:
            raise ValueError(f"LLM no soportado: {self.llm_provider}")  
        
        logger.debug(f"Inicializando DiffClassifier con provider={llm_provider}, model={model}, temperature={temperature}")
        logger.debug(f"Prompt SYSTEM: {self._system}")
    
    # =============================================
    # ==== UTILIDADES
    # =============================================
    
    @staticmethod
    def to_events(diff: DiffRow) -> List[DiffEvent]:
        """Convierte una diferencia de fila en eventos a nivel de columna."""
        events = []
        row_a = diff.row_a
        row_b = diff.row_b
        key = diff.key
        
        logger.trace(f"Convirtiendo DiffRow a eventos: key={key}, row_a={row_a}, row_b={row_b}")
        if row_a is None:  # INSERT
            for col, val in (row_b or {}).items():
                events.append(DiffEvent(
                    key=key, columna=col, valor_a=None, valor_b=val,
                    accion=DiffAction.INSERT
                ))
        elif row_b is None:  # DELETE
            for col, val in (row_a or {}).items():
                events.append(DiffEvent(
                    key=key, columna=col, valor_a=val, valor_b=None,
                    accion=DiffAction.DELETE
                ))
        else:  # UPDATE - solo registramos como eventos las columnas que difieren
            columns = set((row_a or {}).keys()) | set((row_b or {}).keys())
            for col in columns:
                val_a = row_a.get(col)
                val_b = row_b.get(col)
                if val_a != val_b:
                    events.append(DiffEvent(
                        key=key, columna=col,
                        valor_a=val_a, valor_b=val_b
                    ))
        return events

    def _extract_json(self, text: str) -> dict:
        """Extrae y parsea JSON de la respuesta del LLM, gestiona casos de formato conocidos"""
        logger.trace(f"Extrayendo JSON de texto: {text[:300]}{'...' if len(text) > 300 else ''}")
        
        text = text.strip()
        # Eliminar bloques de código markdown
        text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text, flags=re.MULTILINE)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Fallback con regex
            match = re.search(r'(\{[\s\S]*\})', text)
            if match:
                try:
                    return json.loads(match.group(1))
                except:
                    pass
            raise ValueError(f"No se pudo extraer JSON válido. Texto: {text[:300]}...")

    def _build_classification(self, item: DiffRow | DiffEvent, data: dict) -> DiffClassification:
        try:
            logger.debug(f"Construyendo clasificación para item: {item}, con datos: {data}")
            cat_str = data.get("categoria", "UNCERTAIN").upper()
            categoria = DiffCategory[cat_str] if cat_str in DiffCategory.__members__ else DiffCategory.UNCERTAIN

            return DiffClassification(
                key=item.key,
                accion=getattr(item, 'accion', None) or DiffAction.UPDATE,
                categoria=categoria,
                confianza=float(data.get("confianza", 0.0)),
                columnas_afectadas=data.get("columnas_afectadas", []),
                explicacion=data.get("explicacion", ""),
                normalizacion_sugerida=data.get("normalizacion_sugerida",""),
                row_a=getattr(item, 'row_a', None) if isinstance(item, DiffRow) else None,
                row_b=getattr(item, 'row_b', None) if isinstance(item, DiffRow) else None
            )
        except Exception as e:
            logger.error(f"Error construyendo clasificación: {e}")
            return DiffClassification(
                key=item.key,
                accion=DiffAction.UPDATE,
                categoria=DiffCategory.ERROR,
                confianza=0.0,
                columnas_afectadas=[],
                explicacion=f"Error parsing LLM response: {str(e)}",
                row_a=None, row_b=None
            )
            
    # =========== Llamada externa IA ===============
    
    def _haiku_message(self, diff: DiffRow) -> dict:
        """Llamada al modelo Haiku de Anthropic con manejo de reintentos."""
        
        messages = list(FEW_SHOT_EXAMPLES) if self.few_shot else []
        
        prompt = self.prompt_template.format(  
            pk = diff.key,
            source_a=diff.source_a,
            source_b=diff.source_b,
            row_a=json.dumps(diff.row_a, ensure_ascii=False),
            row_b=json.dumps(diff.row_b, ensure_ascii=False),
        )       
        
        messages.append({"role": "user", "content": prompt})
        
        logger.trace(f"Construyendo mensaje para LLM Haiku: {messages}")
        
        return messages
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )   
    def _call_llm_one_row (self, diff: DiffRow) -> dict:
        """Llamada al LLM con reintentos."""
        with timed (logger, f"Llamando al LLM para DiffRow con key={diff.key}", level="DEBUG"):
            try:
                if self.llm_provider == 'anthropic':
                    response = self.client.messages.create(
                        model=self.model,
                        max_tokens=1024,
                        temperature=self.temperature,
                        system = self._system,
                        messages= self._haiku_message(diff)
                    )
                    raw = response.content[0].text.strip()

                elif self.llm_provider == 'ollama':
                    response = self.client.chat(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": self._system},
                            {"role": "user", "content": prompt}
                        ],
                        options={"temperature": self.temperature}
                    )
                    raw = response["message"]["content"].strip()

            
                logger.debug(f"Respuesta LLM:\n{raw[:700]}{'...' if len(raw) > 700 else ''}\n")

                return self._extract_json(raw)

            except Exception as e:
                logger.error(f"Error en llamada LLM: {e}")
                raise
    # ================== Filtro PRE - IA  ======================

    def _cribador(self, row: DiffRow) -> Optional[DiffClassification]:
        """Maneja casos deterministas que no requieren de LLM (INSERT / DELETE)."""
        
        logger.trace(f"Aplicando cribador a DiffRow: key={row.key}, row_a={row.row_a}, row_b={row.row_b}")
        
        if row.row_a is None or row.row_b is None:
            accion = DiffAction.INSERT if row.row_a is None else DiffAction.DELETE
            return DiffClassification(
                key=row.key,
                accion=accion,
                categoria=DiffCategory.DIFFERENT_SEMANTICAL,
                confianza=1.0,
                columnas_afectadas=['*'],
                explicacion=f"{accion.name} del registro con clave {row.key}",
                normalizacion_sugerida=None,
                row_a=row.row_a,
                row_b=row.row_b
            )
        return None    
    # ====================== MÉTODOS PRINCIPALES ======================
    
    def parse_to_diffrows(self, metadata: SegmentStructure, diffs: Iterator[dict]) -> list[DiffRow]    :
       
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

         
        logger.debug(60*'=')
        logger.debug(f"Insertados   (INS) [{len(insert)}] : {insert}"  )
        logger.debug(f"Eliminados   (DEL) [{len(delete)}] : {delete}"  )
        logger.debug(f"Actualizados (UPD) [{len(update)}] : {update}"  )
        logger.debug(60*'-')
        logger.debug(f"Cambios detectados por data-diff en total (2UPD + DEL + INS): {total}")
        logger.debug(60*'=')

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
    
    def classify_one_row(self, row: DiffRow) -> List[DiffClassification]:
        """Clasifica una sola fila (a nivel de columna)."""
        logger.debug(f"Clasificando DiffRow: key={row.key}")
        cribador_result = self._cribador(row)
        if cribador_result:
            return [cribador_result]

        #events = self.to_events(row) TODO: Desarrollar las funcionalidades de eventos
        results = []

        data = self._call_llm_one_row(row)
        classification = self._build_classification(row, data)
        results.append(classification)

        return results

    def classify_row_by_row(self, diffrows: List[DiffRow], max_rows: int = 0) -> List[DiffClassification]:
        """Clasifica múltiples filas de forma síncrona."""
        logger.info(f"Clasificando {len(diffrows)} filas (max_rows={max_rows})")
        all_results = []
        for i, diff in enumerate(diffrows):
            if max_rows > 0 and i >= max_rows:
                break

            try:
                row_results = self.classify_one_row(diff)
                all_results.extend(row_results)
            except Exception as e:
                logger.error(f"Error clasificando fila {diff.key}: {e}")
                # Añadir error como resultado
                all_results.append(DiffClassification(
                    key=diff.key,
                    accion=DiffAction.UPDATE,
                    categoria=DiffCategory.ERROR,
                    confianza=0.0,
                    columnas_afectadas=[],
                    explicacion=f"Excepción: {str(e)}",
                    row_a=diff.row_a,
                    row_b=diff.row_b
                ))

        return all_results

    def report_statistics(self, classifications: List[DiffClassification]):
        """Estadísticas de clasificación."""
        from collections import Counter
        stats = Counter(c.categoria for c in classifications)
        actions = Counter(c.accion for c in classifications)
        falsos_positivos = sum(1 for c in classifications if c.is_false_positive())
        neded_review = sum(1 for c in classifications if c.needs_review())
        total = len(classifications)

        WIDTH = 60
        LABEL = 50   # ancho fijo para etiquetas

        print("\n" + "=" * WIDTH)
        print("REPORTE DE CLASIFICACIÓN".center(WIDTH))
        print("=" * WIDTH)

        print(f"{'Total procesado':<{LABEL}} : {len(classifications):>4d}")

        print("\n--- Clasificación por categoría ---")
        for cat, count in stats.most_common():
            print(f"{cat.name:<{LABEL}} : {count:>4d}")

        print("\n--- Clasificación por acción ---")
        for action, count in actions.most_common():
            print(f"{action.name:<{LABEL}} : {count:>4d}")

        print("\n--- Resumen de Detalles ---")
        print(f"{'Falsos positivos (canonizables/equivalentes)':<{LABEL}} : "
            f"{falsos_positivos:>4d}/{total:<4d} ({falsos_positivos/total:.2%})")

        print(f"{'Requieren revisión (contextual/uncertain/error)':<{LABEL}} : "
            f"{neded_review:>4d}/{total:<4d} ({neded_review/total:.2%})")

        print("=" * WIDTH)
        
    def report_one(self, row:DiffClassification):
        
        """Reporte detallado de una clasificación individual."""
        print("\n" + "=" * 60)
        print(f"Detalle para clave: {row.key}")
        print("=" * 60)
        print(f"Acción: {row.accion.name}")
        print(f"Categoría: {row.categoria.name}")
        print(f"Confianza: {row.confianza:.2f}")
        print(f"Columnas afectadas: {row.columnas_afectadas}")
        print(f"Explicación: {row.explicacion}")
        print(f"Row A: {json.dumps(row.row_a, ensure_ascii=False)}")
        print(f"Row B: {json.dumps(row.row_b, ensure_ascii=False)}")
        print("=" * 60)

    def report_details(self, classifications: List[DiffClassification]):    
        """Reporte detallado de clasificaciones."""
        WIDTH = 60
        LABEL = 50   # ancho fijo para etiquetas

        print("\n" + "=" * WIDTH)
        print("REPORTE DE DETALLES".center(WIDTH))
        print("=" * WIDTH)
        print("\n--- Falsos Positivos ---")       
        [ self.report_one(c)for c in classifications if c.is_false_positive() ]
            
        print("\n--- Requieren revision ---") 
        [ self.report_one(c)for c in classifications if c.needs_review() ]
           