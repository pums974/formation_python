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
print(2+3*9/6)
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

- l’indentation + ":" (utiliser ```pass``` pour un bloc vide)

vocabulaire
-----------
- mutable / immuable
- exception
- shebang ``` #!/usr/bin/env python3 ```
- encoding ``` # -*- coding: utf-8 -*- ```

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
def factorielle(n):                    # définition d'une fonction avec argument
    if n < 2:                          # définition d'un test
        return 1
    else:
        return n * factorielle(n - 1)  # récursivité
```

```
def is_prime(num):
    if num == 1:
        return False
    for i in range(2, int(num ** 0.5)):  # définition d'une boucle avec range
       if (num % i) == 0:
          return False
    return True
```

```
def quadcube(x):
    return x ** 2, x ** 3  # multiples valeurs retours
```

```
def pythagorean_triplets(limit):
    c = 0
    m = 2
    while(c < limit):
        for n in range(1, m + 1):
            a = m * m - n * n
            b = 2 * m * n
            c = m * m + n * n
            if(c > limit):
                break
            if(a == 0 or b == 0 or c == 0):
                break
            print(a, b, c)
        m = m + 1
```

```
def calcul_pi(err, nmax=float("inf")):  # argument optionnel
    n = 0
    erreur = float("inf")
    
    a_n = 1.
    b_n = 2 ** -0.5
    t = 0.25
    while erreur > err and n <= nmax:
        a_np = 0.5 * (a_n + b_n)
        b_np = (a_n * b_n) ** 0.5
        t -= (2 ** n) * (a_n - a_np) ** 2
        
        a_n, b_n = a_np, b_np   # double affectation
        
        erreur = abs(a_n - b_n)
        n = n + 1
        
    pi = (a_n + b_n) ** 2 / (4 * t)
    return pi, n, erreur
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
- generators
  - range

Quelques exemples
-----------------
```
def slow_list_primes(n):
    primes = []  # creation d'une liste vide
    for suspect in range(2,n + 1):
        is_prime = True
        for prime in primes:
            if suspect % prime == 0:
                is_prime = False                
                break
        if is_prime:
            primes.append(suspect)  # aggrandissement de la liste (lent)
    return primes
```
```
def iter_prime(n):
    crible = [False] + [True] * (n - 1)                               # creation d'une liste avec operateurs
    prime = 0
    for is_prime in crible:                                           # parcour d'une liste
        prime += 1
        if is_prime:
            yield prime                                               # générateur
            crible[2*(prime-1)+1::prime] = [False] * (n // prime -1)  # slicing
def list_primes(n):
    return list(iter_prime(n))  # utilisation d'un itérateur
```
```
def fast_list_primes(n):
    crible = [True] * (n // 2)
    for i in range(3, int(n ** 0.5) + 1, 2):
        if crible[i // 2]:
            crible[i * i // 2::i] = [False] * ((n - i * i - 1) // (2 * i) + 1)
    return [2] + [2 * i + 1 for i in range(1, n // 2) if crible[i]]  # comprehension de liste
```

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
	f.write("{:>10} est du texte formaté".format("ceci"))
```
  - **toujours** decode en entrée
  - **toujours** encode en sortie
  - formatage
    - ```print("un nombre : %d" % nombre)```
    - ```print("un nombre : {:}".format(nombre))```
    - ```print(f"un nombre : {nombre:}")``` (uniquement en python 3.6)
- autres fichiers -> utiliser un paquets appropié (pillow pour les images) -> ecosystème

Modules
-------
- ```__init__.py```
- import _ / from _ import _ / from _ import *
- import implique execution !
- peut être importé
  - tout module dans sys.path (alimenté par PYTHONPATH)
  - tout sous module avec spam.egg (importera spam puis egg dans le dossier spam)

apercu de la stdlib
-------------------
- sys et os
  deux modules très communs pour lire et agire sur l'environnement du code
- shutil
- math / cmath
  contiens les fonction mathematiques classiques (reelles / complexes)
- copy
  contiens en particulier deepcopy
- pathlib
  permet de manipuler proprement des chemin de fichier (mieux que dossier + "/" + fichier)
- time
- ...

Quelques exemples
-----------------

Classes (light)
---------------
**En python TOUT est objet**
```
class egg(object):
  total_number = 0
  
  def __init__(self,number = 1):
    self.number = number
    egg.total_number += number
    
  @classmethod
  def depuis_recette(cls,ingredients):
    return cls(ingredients["oeufs"])
    
  def __del__(self):
    egg.total_number -= self.number
    
  def combien(self):
    return self.number

  @staticmethod
  def combien_egg():
    return egg.total_number
```
- Methodes speciales
  - ```__init__```
  - ```__del__```
  - ```__str__``` et ```__repr__```
  - les operateurs ```__mul__``` ...
  - ...
- @classmethod et @staticmethod

variables ++
------------
- reference partagé (faire un beau dessin ou une belle animation)
- portée des variables
  - **L**ocal
  - **E**nglobante
  - **G**lobale
  - **B**uiltins
- espace de nommage
  - fonction
  - instance
  - classe
  - module
- heritage multiple et MRO

