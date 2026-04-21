

[[TOC]]

#  Ollama en docker 

## Instalación 
curl -fsSL https://ollama.com/install.sh | sh

## levantar modelo
docker exec -it ollama ollama serve &   

# Verificar que el servidor está corriendo
  
 ``` 
curl -s http://localhost:11434/api/tags | jq 
{
  "models": [
    {
      "name": "qwen2.5:7b",
      "model": "qwen2.5:7b",
      "modified_at": "2026-04-17T19:27:40.200724826Z",
      "size": 4683087332,
      "digest": "845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e",
      "details": {
        "parent_model": "",
        "format": "gguf",
        "family": "qwen2",
        "families": [
          "qwen2"
        ],
        "parameter_size": "7.6B",
        "quantization_level": "Q4_K_M"
      }
    }
  ]
}

docker exec -it ollama ollama list  
NAME          ID              SIZE      MODIFIED      
qwen2.5:7b    845dbda0ea48    4.7 GB    7 minutes ago    
```
## Instalar librerias python 
pip install --break-system-packages ollama

## Motores validados

- gemma2:2b : pérdida de foco de tarea
- qwen2.5:7b
- qwen3:8b

## **Descargar** el modelo

```bash
sudo docker exec -e OLLAMA_SKIP_VERIFY=true -it ollama ollama pull gemma2:2b
```

Si da errores de timeout , activar VPN (inspección de paquetes):

```bash
pulling manifest

Error: max retries exceeded: Get "<https://dd20bb891979d25aebc8bec07b2b3bbc.r2.cloudflarestorage.com/ollama/docker/registry/v2/blobs/sha256/74/7462734796d67c40ecec2ca98eddf970e171dbb6b370e43fd633ee75b69abe1b/data?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=66040c77ac1b787c3af820529859349a%2F20260411%2Fauto%2Fs3%2Faws4_request&X-Amz-Date=20260411T201246Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=55d7d8df0bf624ced63e7b719e0ecc917cce2c9045c47e660885a3fd8485a459>": tls: failed to verify certificate: x509: certificate is not valid for any names, but wanted to match dd20bb891979d25aebc8bec07b2b3bbc.r2.cloudflarestorage.com
```

Comprobación: (en el ejemplo aparece **core1.netops.test** en lugar de cloudfarestorage como nodo que sirve el certificado!!!)

```bash
$ openssl s_client -connect r2.cloudflarestorage.com:443 -servername r2.cloudflarestorage.com
CONNECTED(00000003)
depth=0 C = EU, ST = SOME-ST, L = SOME-CITY, O = Widgits Pty Ltd, OU = Packetland, CN = **core1.netops.test**, emailAddress = admin@Packetland
verify error:num=18:self-signed certificate
verify return:1
depth=0 C = EU, ST = SOME-ST, L = SOME-CITY, O = Widgits Pty Ltd, OU = Packetland, CN = core1.netops.test, emailAddress = admin@Packetland
verify return:1
---

Certificate chain
 0 s:C = EU, ST = SOME-ST, L = SOME-CITY, O = Widgits Pty Ltd, OU = Packetland, CN = core1.netops.test, emailAddress = admin@Packetland
   i:C = EU, ST = SOME-ST, L = SOME-CITY, O = Widgits Pty Ltd, OU = Packetland, CN = core1.netops.test, emailAddress = admin@Packetland
   a:PKEY: rsaEncryption, 2048 (bit); sigalg: RSA-SHA256
   v:NotBefore: Feb 19 14:46:27 2026 GMT; NotAfter: Sep 18 14:46:27 2124 GMT
---

Server certificate
-----BEGIN CERTIFICATE-----
...
-----END CERTIFICATE-----
subject=C = EU, ST = SOME-ST, L = SOME-CITY, O = Widgits Pty Ltd, OU = Packetland, CN = core1.netops.test, emailAddress = admin@Packetland
issuer=C = EU, ST = SOME-ST, L = SOME-CITY, O = Widgits Pty Ltd, OU = Packetland, CN = core1.netops.test, emailAddress = admin@Packetland
---
```

## Ejecutar el modelo

Aunque el proceso ollama esté arriba hay realizar un **run** con el modelo que queremos para que el proceso ollama atienda peticiones:

```bash
docker exec -it ollama ollama run gemma2:2b
```

## Problemas clasificación

El **prompt** bajo QWEN 2.5

```
Eres un experto en reconciliación de datos.

### SCHEMA (Titanic)
- Primary key: PassengerId (es la primera clave en cada diccionario)
- Columnas: survival, pclass, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked

Te voy a proporcionar dos datasets con el mismo schema.

### TAREA (sigue estos pasos exactamente en orden):
1. Compara fila por fila usando **PassengerId** como clave.
2. Para cada PassengerId decide **SOLO** una de estas tres categorías:
   - "diferente" → hay una diferencia real en el valor
   - "equivalente" → los valores son iguales o solo difieren en formato/tipo
   - "inconsistencia_normalizacion" → es un artefacto de normalización

### REQUISITOS DE SALIDA (OBLIGATORIOS):
Devuelve **ÚNICAMENTE** un array JSON válido y nada más.
No uses markdown, no agregues texto explicativo, no uses ```json, no pongas nada antes ni después del array.

El formato exacto debe ser:

  {{
    "categoria": "diferente|equivalente|inconsistencia_normalizacion",
    "confianza": 0.XX,
    "columnas_afectadas": ["col1", "col2"],
    "explicacion": "breve explicación en español",
    "normalizacion_sugerida": "SQL o null"
  }},
```
**Retorna** tras

  * Total (ms): 490345.357359
  * Prompt eval (ms): 384283.820756
  * Generation (ms): 101926.914256

```
Based on the information provided in the two datasets, here are some observations and comparisons:

### Common Attributes:
- Both datasets contain a `PassengerId` field.
- Both have fields for `Survived`, `Pclass`, `Name`, `Sex`, `Age`, `SibSp`, `Parch`, `Ticket`, `Fare`, `Cabin`, and `Embarked`.

### Differences in Data:
1. **Number of Passengers:**
   - Dataset A has 926 passengers.
   - Dataset B has 925 passengers.

2. **Missing Values:**
   - In Dataset A, there are missing values for the `Age` field (indicated by "NA").
   - In Dataset B, some fields like `Cabin` and `Fare` have missing or inconsistent data (e.g., "0.000" in `Fare`, "LINE" in `Ticket`).

3. **Unique Entries:**
   - There are unique entries between the two datasets:
     - In Dataset A, there is a passenger with `PassengerId` 123456789 (not present in B).
     - In Dataset B, there is a passenger with `PassengerId` 912 that has different details compared to the same ID in A.

### Example Comparisons:
- **Passenger 1:**
  - **Dataset A:** 
    - PassengerId: 1
    - Survived: 0
    - Pclass: 3
    - Name: Braund, Mr. Owen Harris
    - Sex: male
    - Age: 22
    - SibSp: 1
    - Parch: 0
    - Ticket: A/5 21171
    - Fare: 7.25
    - Cabin: unknown
    - Embarked: Southampton

  - **Dataset B:** 
    - PassengerId: 912
    - Survived: 0
    - Pclass: 1
    - Name: Carlsson, Mr. Frans Olof
    - Sex: male
    - Age: 33
    - SibSp: 0
    - Parch: 0
    - Ticket: 695
    - Fare: 5.000
    - Cabin: B51 B53 B55
    - Embarked: Southampton

### Summary:
- The datasets share many common attributes but differ in the number of passengers and some specific details.
- There are discrepancies in passenger information, particularly for `Age`, `Fare`, `Cabin`, and `Ticket` fields.

If you need to merge or compare these datasets further, it would be helpful to identify a unique key (like `PassengerId`) and handle missing values appropriately.
```
o 
```

Based on the information provided in the two datasets, here are some observations and comparisons:

### Common Attributes:
- Both datasets contain a `PassengerId`, `Survived`, `Pclass`, `Name`, `Sex`, `Age`, `SibSp`, `Parch`, `Ticket`, `Fare`, `Cabin`, and `Embarked` attribute.

### Differences in Data:
1. **Number of Passengers:**
   - Dataset A has 926 passengers.
   - Dataset B has 925 passengers.

2. **Missing Values:**
   - In Dataset A, there are missing values for the `Cabin` column (indicated by "nan").
   - In Dataset B, some attributes like `Age`, `SibSp`, and `Parch` have missing values represented as `null`.

3. **Unique Attributes:**
   - The only unique attribute between the two datasets is the `PassengerId`. Each dataset has a different set of `PassengerId`s.

4. **Survival Rates:**
   - Dataset A shows that 502 passengers survived (54%).
   - Dataset B also shows that 502 passengers survived (54%).

### Observations:
- The survival rates are the same in both datasets, which is a good consistency check.
- There are slight differences in the number of records and some missing values.

### Example Comparisons:
1. **Passenger with ID 892:**
   - Dataset A: 
     ```
     PassengerId: 892
     Survived: 0
     Pclass: 3
     Name: "Braund, Mr. Owen Harris"
     Sex: "male"
     Age: 22.0
     SibSp: 1
     Parch: 0
     Ticket: A/5 21171
     Fare: 7.25
     Cabin: nan
     Embarked: "S"
     ```
   - Dataset B:
     ```
     PassengerId: 892
     Survived: 0
     Pclass: 3
     Name: "Braund, Mr. Owen Harris"
     Sex: "male"
     Age: null
     SibSp: 1
     Parch: 0
     Ticket: A/5 21171
     Fare: 7.25
     Cabin: nan
     Embarked: "S"
     ```

2. **Passenger with ID 894:**
   - Dataset A:
     ```
     PassengerId: 894
     Survived: 0
     Pclass: 3
     Name: "Allen, Mr. William Henry"
     Sex: "male"
     Age: 35.0
     SibSp: 0
     Parch: 0
     Ticket: 373450
     Fare: 8.05
     Cabin: nan
     Embarked: "S"
     ```
   - Dataset B:
     ```
     PassengerId: 894
     Survived: 0
     Pclass: 3
     Name: "Allen, Mr. William Henry"
     Sex: "male"
     Age: null
     SibSp: 0
     Parch: 0
     Ticket: 373450
     Fare: 8.05
     Cabin: nan
     Embarked: "S"
     ```

### Summary:
- Both datasets are consistent in their overall structure and the majority of attributes.
- There are minor differences such as missing values represented differently (e.g., `nan` vs `null`) and a slight difference in the number of records.

These observations can help in understanding the data better and ensuring that any analysis or model training is done on clean, consistent datasets.
Total (ms): 525643.569562
Prompt eval (ms): 382445.598316
Generation (ms): 138768.884381
``` 
o 
``` 
Based on the information provided in both datasets, here are some observations and comparisons:

### 1. **Survival Rates by Class:**
   - **First Class (Pclass = 1):**
     - Dataset A: 62% survival rate.
     - Dataset B: 57% survival rate.

   - **Second Class (Pclass = 2):**
     - Dataset A: 47% survival rate.
     - Dataset B: 38% survival rate.

   - **Third Class (Pclass = 3):**
     - Dataset A: 24% survival rate.
     - Dataset B: 19% survival rate.

### 2. **Survival Rates by Gender:**
   - **Male Survival Rate:**
     - Dataset A: 18%.
     - Dataset B: 17%.

   - **Female Survival Rate:**
     - Dataset A: 70%.
     - Dataset B: 63%.

### 3. **Survival Rates by Age:**
   - **Children (Age < 14):**
     - Dataset A: 52% survival rate.
     - Dataset B: 48%.

   - **Adults (Age ≥ 14):**
     - Dataset A: 20% survival rate.
     - Dataset B: 19%.

### 4. **Survival Rates by Embarked Port:**
   - **Port C (Cherbourg):**
     - Dataset A: 53% survival rate.
     - Dataset B: 56%.

   - **Port Q (Queenstown):**
     - Dataset A: 27% survival rate.
     - Dataset B: 24%.

   - **Port S (Southampton):**
     - Dataset A: 19% survival rate.
     - Dataset B: 18%.

### 5. **Survival Rates by Cabin Presence:**
   - **With Cabin:**
     - Dataset A: 36% survival rate.
     - Dataset B: 34%.

   - **Without Cabin:**
     - Dataset A: 20% survival rate.
     - Dataset B: 19%.

### 6. **Survival Rates by Ticket Class and Gender:**
   - **First Class, Female:**
     - Dataset A: 87% survival rate.
     - Dataset B: 84%.

   - **Second Class, Male:**
     - Dataset A: 25% survival rate.
     - Dataset B: 19%.

### 7. **Survival Rates by Ticket Class and Age:**
   - **First Class, Children (Age < 14):**
     - Dataset A: 63% survival rate.
     - Dataset B: 60%.

   - **Third Class, Adults (Age ≥ 14):**
     - Dataset A: 22% survival rate.
     - Dataset B: 18%.

### Summary:
- Both datasets show a higher survival rate for women and children compared to men and adults.
- The first class had the highest survival rates in both datasets, followed by second class, then third class.
- The survival rates are slightly lower in Dataset B compared to Dataset A across all categories.

These comparisons can help identify trends and patterns that might be useful for further analysis or modeling.
```
o

```

Based on the information provided in the two datasets, here are some observations and comparisons:

1. Both datasets contain passenger data with similar attributes such as PassengerId, Survived (in Dataset B), Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, and Embarked.

2. The number of passengers in each dataset is different:
   - Dataset A has 1309 entries
   - Dataset B has 891 entries

3. Some passengers appear in both datasets with the same attributes (e.g., PassengerId: 6, 470, 549).

4. There are discrepancies between the two datasets:
   - Passenger 2 (Dataset A) does not exist in Dataset B
   - Passengers 1 and 891 (Dataset B) do not exist in Dataset A

5. The survival rate is different due to the difference in sample size and passengers included:
   - In Dataset A, there are no explicit survival labels, but we can infer that about 38% survived based on the number of survivors (476 out of 1224).
   - In Dataset B, approximately 34% survived (295 out of 889).

6. The age distribution seems to be similar in both datasets with a range from 0.42 to 80 years.

7. There are differences in the number of passengers per class:
   - In Dataset A: Pclass 1 has 314, Pclass 2 has 285, and Pclass 3 has 710.
   - In Dataset B: Pclass 1 has 119, Pclass 2 has 185, and Pclass 3 has 587.

These observations highlight the differences between the two datasets and suggest that they may have been collected or curated from different sources.
Total (ms): 514144.670483
Prompt eval (ms): 430580.072803
Generation (ms): 82379.453488
None
```
o
```
{
    "id" : "916",
    "categoria": "FALSO_POSITIVO_NORMALIZACION",
    "confianza": 0.85,
    "columnas_afectadas": ["Sex", "Age", "Pclass"],
    "explicacion": "El pasajero es un hombre joven de clase alta, lo cual no es consistente con la edad y el sexo.",
    "normalizacion_sugerida": "WHERE Sex = 'male' AND Age > 30 AND Pclass = 1"
}
´´´

