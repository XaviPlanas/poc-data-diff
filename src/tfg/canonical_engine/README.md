# Reglas de canonización
| ID | Regla | Compatibilidad | Relevancia | Pérdida_datos | Metadato_adicional |
|----|------|---------------|------------|---------------|--------------------|
| 1 | Cast boolean a entero 0/1 | 9 | 10 | 0 | Sí |
| 2 | Cast boolean a string 'true'/'false' | 8 | 9 | 0 | No |
| 3 | Normalizar TINYINT/BIT a BOOLEAN | 10 | 9 | 0 | Sí |
| 4 | TIMESTAMP a UTC | 10 | 10 | 1 | Sí |
| 5 | Fechas a ISO 8601 string | 9 | 10 | 2 | Sí |
| 6 | DATE a YYYY-MM-DD | 9 | 8 | 0 | No |
| 7 | Unificar precisión TIMESTAMP | 7 | 8 | 3 | Sí |
| 8 | YEAR a INTEGER | 8 | 7 | 0 | No |
| 9 | INTERVAL a ISO 8601 duration | 6 | 7 | 1 | Sí |
| 10 | Enteros a BIGINT | 9 | 9 | 0 | Sí |
| 11 | DECIMAL a precisión fija | 8 | 8 | 4 | Sí |
| 12 | FLOAT a DOUBLE | 7 | 7 | 5 | No |
| 13 | UNSIGNED a signed | 8 | 6 | 2 | Sí |
| 14 | BIT a entero | 9 | 8 | 0 | No |
| 15 | TRIM y normalizar whitespace | 8 | 9 | 1 | Sí |
| 16 | LOWERCASE | 9 | 8 | 6 | Sí |
| 17 | Normalizar collation | 7 | 9 | 5 | Sí |
| 18 | '' a NULL | 10 | 9 | 3 | Sí |
| 19 | Encoding a UTF-8 | 6 | 7 | 4 | Sí |
| 20 | UUID a string lowercase | 9 | 9 | 0 | No |
| 21 | JSON canonizado | 7 | 9 | 2 | Sí |
| 22 | ARRAY a JSON ordenado | 6 | 8 | 1 | Sí |
| 23 | BINARY a HEX | 8 | 7 | 0 | No |
| 24 | ENUM a string | 9 | 8 | 0 | Sí |
| 25 | GEOMETRY a WKT | 5 | 6 | 2 | Sí |
| 26 | XML canonizado | 6 | 5 | 3 | Sí |
| 27 | COALESCE NULL | 8 | 4 | 7 | Sí |
| 28 | Hash fila pre-canonización | 10 | 10 | 0 | Sí |
| 29 | Normalizar comillas/escaping | 7 | 6 | 1 | No |
| 30 | Columnas a snake_case | 9 | 8 | 0 | Sí |

**TODO:** Añadir columna família de la regla

# Tabla compatibilidades regla/motores ( y expresiones SQL equivalentes adoptadas )