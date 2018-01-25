Avancé
======

Pratique
--------
- utilisation de décorateur
  ```
  class NbAppel(object):
    def __init__(self, f):
      self.appels = 0
      self.f = f
    def __call__(self,*args, **xargs):
      self.appels += 1
      print("{} a été appelé {} fois".format(self.f.__name__, self.appels))
      return self.f(*args, **xargs)
  ```
  ```
  def PrintAppel(f):
    def new_f(*args, **xargs):
      new_f.NbAppels += 1
      print("On rentre dans {}".format(f.__name__))
      res = f(*args, **xargs)
      print("On sort de {}".format(f.__name__))
      return res
    new_f.NbAppels = 0
    return new_f
  ```
  
- pprint
- underscore
  - separator in number ``` 10_000 ``` (only python 3.6)
  - last result in the interpreter ``` _ ```
  - I don't care ``` _ = f() ``` (dangerous with internationnaization)
  - weakly private ``` _something ``` (won't be imported with ```import *```)
  - avoid conflict ``` list_ ```
  - more private (mangled) ```__stuff```
  - magic methods (also mangled) ``` __init__```
  - for internationnalization ```_()```
- Tests faux
  - ```None```
  - ```0```
  - ```__nonzero__()```
  - ```__len__()```
- a is b : a et b pointent vers le même objet
- unpacking dans les boucles 
  - zip ``` for i, j in zip(list1,list2):```
  - enumerate ``` for index, elem in list1:```
- utiliser collections.defaultdict(list) pour initialiser un dictionnaire de listes
- try is very fast buit except is very slow
- **never** use a mutable optionnal arg
- use a shallow copy to modify a list in a loop (or be very carefull)

  ```
  for valeur in copy(ensemble):
    if 'bert' not in valeur:
      ensemble.discard(valeur)
  ```
- mecanisme d'import
  - Chercher si le module os existe.
  - Chercher si le module a déjà été importé. Si oui, s’arrêter ici
  - Si non, chercher si il a été déjà compilé en .pyc.
  - Si ce n’est pas le cas, compiler le fichier .py en .pyc.
  - Executer le module 


  

Packaging
---------
- PEP (for prettyness)
- type hinting
- docstring (auto-documentation)
- pytest  (unit-testing)
- gettext (auto-internationnalization)

performance
-----------
- profiling
- numexpr
- parallelism
  - asyncio
  - multithreading
  - multiprocessing
  - mpi
- compiler
  - f2py
  - cython
  - numba
