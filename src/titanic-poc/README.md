== Comparar dataset titanic entre mysql y postgresql
(venv) xavi@dragoneta:~/poc-data-diff$ data-diff  mysql://poc:poc3306@localhost/pocdb titanic postgresql://poc:poc5432@localhost/tfgdb titanic_modified -k PassengerId -d --stats
"""
891 rows in table A
891 rows in table B
0 rows exclusive to table A (not present in B)
0 rows exclusive to table B (not present in A)
0 rows updated
891 rows unchanged
0.00% difference score

Extra-Info:
  rows_downloaded = 891
  """
data-diff  mysql://poc:poc3306@localhost/pocdb titanic postgresql://poc:poc5432@localhost/tfgdb titanic -k PassengerId -c % 


== diagrama mermaid

