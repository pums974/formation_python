Les bases
========

Historique
----------
- 1989 : CWI : **Guido van Rossum**
- 1991 : premiere version publique
- 1995 : CNRI
- 1999 : Computer Programming for Everybody (CNRI + DARPA) objectif : python pour l’**enseignement** de la programation
- 2001 : Python software fundation
- 2008 : version 3
- 2013 : langage enseigné en prépa
- 2018 : 4eme langage le plus populaire (après Java, C, C++)
- 2020 : mort de python 2.x

installation
------------
- Windows : anaconda + conda + pip
- linux : déjà installé + apt + pip --user
- mac : déjà installé + brew + pip --user

interface
---------
vim+terminal / IDLE / spyder / notebook

Helloworld
----------
```
print("hello world")
```
c’est aussi simple que ca !

Syntaxe (cheatsheet)
--------------------
- Les variables :
  - noms
    - commence par une lettre
    - casse importante
    - pas de mots clefs
    - pas de symboles opérateurs
    - mais les accents sont autorisés (à éviter quand même)
  - typage dynamique
    - types numériques
      - entiers de précision infinie
      - float de double précision
      - complex
    - types « itérables »
      - tuple
      - list
      - set
      - dist
      - str
      - frozenset
      - bytarray
- les operateurs

  ``` + - * ** / // % = < > <= >= == != () {} [] ‘ " ```
  
- les mots clefs : 

```
from keyworld import kwlist
print(kwlist)
```

- l’indentation + ":"

vocabulaire
-----------
- mutable / immuable
- exception

Aides
-----
- help
- dir
- internet
  - docs.python.org
  - stackoverflow
  - reddit
  - ...
  
quelques exemples
-----------------
```
def factorielle(n):
    if n < 2:
        return 1
    else:
        return n * factorielle(n - 1)
```

listes, dicts et compagnie
--------------------------
- types courants
  - tuple : (a,b,c)
    - immutable
    - indexable avec des entiers
    - parenthèses optionelles ( a,b = c,d )
  - list : [a, b, c]
    - mutable
    - indexable avec des entiers
  - set : {a, b, c}
    - mutable
    - indexable avec des entiers
    - pas de doublons
  - dict : {a:b, c:d}
    - mutable
    - indexation avec a et c
    - a et c sont des cléfs uniques et immutables
  - str : "spam" == 'spam' == """spam"""
    - immutable
    - indexable avec des entiers
    - can contains only decoded characters
    - unicode par default
- type rares
  - frozenset
    - immutable
    - indexable avec des entiers
    - pas de doublons
  - bytearray
    - mutable
    - indexable avec des entiers
    - can contains only encoded characters
- slicing (pour indexation avec des entiers)
  - A[1] : deuxième élément
  - A[-1] : dernier élément
  - A[4:8:2] : 5eme et 7eme éléments 
- methodes
  - len
  - append
  - sort / sorted
- comprehension
  ```[x**2 for x in range(10)]```

Quelques exemples
-----------------
```
a = []
for i in range(n):
    if i % 2 != 0:
        for j in range(n):
            if j % 3 != 0:
                a.append(i + j if i != j else 0)
```

Classes (light)
---------------
- Juste pour le principe
- evoquer les methodes speciales

Modules
-------
- __init__.py
- import _ / from _ import _ / from _ import *
- import implique execution !

variables ++
------------
- reference partagé
- portée des variables
- espace de nommage

I/O
---
- Fichier texte
  - lecture
```
for line in open(input) :
	print(line.split())
```
  - ecriture
```
with open(output) as f :
	f.write("{:>10} du texte formaté".format("ceci"))
```

autres fichiers -> utiliser un paquets appropié (pillow pour les images) -> ecosystème

apercu de la stdlib
-------------------
- sys
- os
- ...

Quelques exemples
-----------------

ecosysteme
==========
Rappel pip
----------
Numpy (+scipy)
--------------
Matpotlib
---------
Quelques exemples
-----------------
simpy
-----
Opencv
------
pandas
------
pillow
------
.... ?

Avancé
======
Packaging
---------
- PEP
- docstring
- pytest

performance
-----------
- profiling
- parallelism
  - asyncio
  - multithreading
  - multiprocessing
  - mpi
- compiler
  - f2py
  - cython
  - numba
