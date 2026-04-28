-- Crear un patrón reproducible de cambios
update titanic_modified tm set "Name" = concat(tm."Name", '_mod5') where tm."PassengerId"::text like '%5'
--select concat(tm."PassengerId",'_mod5') from titanic_modified tm where tm."PassengerId"::text like '%5'  

-- Crear cambios en múltiples columnas
UPDATE titanic_modified tm
SET 
    "Survived" = 1 - tm."Survived",
    "Parch" = 10,
    "SibSp" = 10
WHERE "PassengerId" IN (321, 817, 743);


-- Modificar datos semánticamente idénticos
UPDATE public.titanic_modified
	SET "Name"='Braund, Sr. Owen Harris'
	WHERE "PassengerId"=1;
UPDATE public.titanic_modified
	SET "Name"='Cumings, Sra. John Bradley (Florence Briggs Thayer)'
	WHERE "PassengerId"=2;
UPDATE public.titanic_modified
	SET "Name"='Heikkinen, Sra. Laina'
	WHERE "PassengerId"=3;
UPDATE public.titanic_modified
	SET "Name"='Futrelle, Jacques Heath (Lily May Peel)'
	WHERE "PassengerId"=4;
UPDATE public.titanic_modified
	SET "Name"='Moran, Sr. James'
	WHERE "PassengerId"=6;
UPDATE public.titanic_modified
	SET "Name"='McCarthy, Sr. Timothy J'
	WHERE "PassengerId"=7;
UPDATE public.titanic_modified
	SET "Name"='Palsson, Master Gosta Leonard'
	WHERE "PassengerId"=8;
UPDATE public.titanic_modified
	SET "Name"='Johnson Mrs Oscar W. (Elisabeth Vilhelmina Berg)'
	WHERE "PassengerId"=9;
UPDATE public.titanic_modified
	SET "Name"='Nasser, Nicholas (Adele Achem)'
	WHERE "PassengerId"=10;
UPDATE public.titanic_modified
	SET "Name"='Sandstrom Miss. Marguerite Rut'
	WHERE "PassengerId"=11;
