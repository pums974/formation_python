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

La force de Python (scipy)
------------------

- **Batteries included**
- **Easy to learn**
- **Easy
communication**
- **Efficient code**
- **Universal**

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

```python
print("Hello World !")
```

```python
print(2 + 3 * 9 / 6)
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
  ``` + - * ** / // % = < > <= >= == != () {} [] ‘ " @ . ```  
- l’indentation + ":" (utiliser ```pass``` pour un bloc vide)
- les mots clefs :

```python
from keyword import kwlist
print(kwlist)
```

vocabulaire
-----------
- mutable / immuable
  un objet mutable peux changer contrairement à un objet immuable    
  les types numériques sont évidement immuable  
    

- exception  
  python ne plante jamais, quoi que vous écriviez (ou presque jamais)  
  si un problème apparait, python lance une exception qui arrete le programme (sauf si elle est traitée)
  
  
- shebang ```
#!/usr/bin/env python3 ```
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

```python
def factorielle(n):                    # définition d'une fonction avec argument
    if n < 2:                          # définition d'un test
        return 1                       # multiples points de retours
    else:
        return n * factorielle(n - 1)  # récursivité
    
res = factorielle(5)
print(res)
```

```python
def factorielle_cinq():                # définition d'une fonction sans argument
    return factorielle(5)

res = factorielle_cinq()
print(res)
```

```python
def is_prime(num):
    if num == 1:
        return False
    for i in range(2, int(num ** 0.5)+1):  # définition d'une boucle avec range
       if (num % i) == 0:
          return False
    return True

print(is_prime(7))                         # enchainement de fonction
print(is_prime(9))
```

```python
def quadcube(x):
    return x ** 2, x ** 3  # multiples valeurs retours

x1, x2 = quadcube(7)
print(x1,x2)
```

```python
def pythagorean_triplets(limit):
    """ print all pythagorean triplets below a limit
    
    A pythagorean triplet is a triplet of integers a, b and c such that
    a^2 + b^2 = c^2
    """                                       # use a docstring
    c = 0
    m = 2
    while(c < limit):
        for n in range(1, m + 1):
            a = m * m - n * n
            b = 2 * m * n
            c = m * m + n * n
            if(c > limit):
                break                         # arret d'une boucle
            if(a == 0 or b == 0 or c == 0):
                break
            print(a, b, c)
        m = m + 1
        
help(pythagorean_triplets)
pythagorean_triplets(30)
```

```python
def calcul_pi(err, nmax=float("inf")):    # argument optionnel
    n = 0
    erreur = float("inf")
    
    a_n = 1.
    b_n = 2 ** -0.5
    t = 0.25
    while erreur > err and n < nmax:
        a_np = 0.5 * (a_n + b_n)
        b_np = (a_n * b_n) ** 0.5
        t -= (2 ** n) * (a_n - a_np) ** 2  # soustraction sur place
        
        a_n, b_n = a_np, b_np              # double affectation
        
        erreur = abs(a_n - b_n)
        n = n + 1
        
    pi = (a_n + b_n) ** 2 / (4 * t)
    return pi, n, erreur

print(calcul_pi(1e-15))
print(calcul_pi(1e-15, 3))
print(calcul_pi(1e-15, nmax=2))
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
- generators / iterator
  - range
  - enumerate
  - zip
  - open("filename")
  - ...

Quelques exemples
-----------------

```python
def slow_list_primes(n):
    primes = []                      # creation d'une liste vide
    for suspect in range(2,n + 1):
        is_prime = True
        for prime in primes:
            if suspect % prime == 0:
                is_prime = False                
                break
        if is_prime:
            primes.append(suspect)   # aggrandissement de la liste (lent)
    return primes

slow_list_primes(10)
```

```python
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

list_primes(20)
```

```python
def fast_list_primes(n):
    crible = [True] * (n // 2)                                       # division entière
    for i in range(3, int(n ** 0.5) + 1, 2):
        if crible[i // 2]:
            crible[i * i // 2::i] = [False] * ((n - i * i - 1) // (2 * i) + 1)
    return [2] + [2 * i + 1 for i in range(1, n // 2) if crible[i]]  # comprehension de liste

list_primes(50)
```

I/O
---
- **toujours** decode en entrée
- **toujours** encode en sortie
-
formatage
  - ```print("un nombre : %d" % nombre)```
  - ```print("un nombre :
{:}".format(nombre))```
  - ```print(f"un nombre : {nombre:}")``` (uniquement en
python 3.6)
- En général : utiliser un paquets appropié (pillow pour les images) -> ecosystème
- Operation sur les string
  - TODO
- Fichier texte
  - lecture

```python
for line in open("README.md", encoding="utf-8"):      # use of encoding strongly encouraged
    print(line.split())
```

- ecriture

```python
with open("output_filenamme", 'w', encoding="utf-8") as f:  # utilisation d'un contexte
    f.write("{:>10} est du texte formaté".format("ceci"))
```

VOUS ÊTES PRÈT A PYTHONISER
===========================

Quelques éléments supplémentaire à garder en tête
-------------------------------------------------

Modules
-------
-
```__init__.py```
- differentes facon d'importer
  - ```import paquet```
  Puis paquet.fonction()
  - ```from paquet import fonction```  
  Puis fonction()
- ```from paquet import *```  
  Puis fonction()  
  A évité !
  
  
- import implique execution !  
  ```if __name__ == '__main__'```
- peut être importé
  - tout module dans sys.path (alimenté par PYTHONPATH)
  - tout sous module avec spam.egg (importera spam puis egg dans le dossier spam)

```python
from sys import path
print(path)
```

apercu de la stdlib
-------------------
- builtins     : est toujours importé automatiquement, contiens la base (int, print(), ...)
- sys et os    : deux modules très communs pour lire et agir sur l'environnement du code
- shutil       : permet de manipuler des fichier (copy, renomage, ...)
- math / cmath : contiens les fonction mathematiques classiques (reelles / complexes)
- copy         : contiens en particulier deepcopy
- pathlib      : permet de manipuler proprement des chemin de fichier (mieux que dossier + "/" + fichier)
- time
- ...

Classes (light)
---------------
**En python TOUT est objet**

```python
class simple(object):                         # Tous les objets heritent d'object
  def __init__(self, attribut="value"):       # constructeur
    self.attribut = attribut                  # Bonne facon de définir les attributs
    
  def get_attribut(self):                     # fonction normale de l'instance
    return self.attribut
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
- portée des variables (lecture)
  - **L**ocal
- **E**nglobante
  - **G**lobale
  - **I**mports
- espace de nommage (ecriture)
- fonction
  - instance
  - classe
  - module
- heritage multiple et MRO

Bonnes pratiques
================

La philosophie de python

```python
import this # PEP 20
```

- Respecter la [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- 4 espaces pas de tabulation
- Partir de l'existant avant de coder un algo
- un code clair sans commentaire est mieux qu'un code obscure mais commenté
- utiliser des docstrings

- [The Hitchhiker’s Guide to Python!](http://docs.python-guide.org/en/latest/)
- http://www.scipy-lectures.org
