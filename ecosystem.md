ecosysteme
==========

Rappel pip
----------
- Windows
  - conda install package
  - pip install package
- Linux et Mac
  - pip install --user package (linux et mac)
  
  
Numpy (+scipy)
--------------
http://mathesaurus.sourceforge.net/matlab-python-xref.pdf

Numpy :
- ```ndarray``` : un tableaux d'un dtype unique **beaucoup** plus performants que les listes python
- **mutable**
- ```dtype``` : type en numpy
- indexation et slicing : possibilité d'utiliser
  - une liste d'entier par dimension
  - un tableau d'index
  - un masque
- ```shape``` : forme d'un tableau
- creation :
  - ```empty``` : creation rapide mais sans initialisation
  - ```zeros```, ```zeros_like```, ```ones```, ```ones_like``` : creation avec initialisation à 0 ou 1
  - ```arange(debut, fin, pas)```
  - ```linspace(debut, fin, nb de points)```
  - ```meshgrid``` : maillage
- produit matriciel : ```@```
- **Beaucoup** de paquets utilisent cette structure de donnée
- utiliser ```&``` et ```|``` au lieu de ```and``` et ```or```
- quelques fonctions utiles  (survol)
  - ```linalg.eigvals``` : valeurs propres
  - ```linalg.det``` : determinant
  - ```linalg.solve``` : resoud Ax = b
  - ```linalg.inv``` : inverse une matrice
  - ```sort``` : tri
  - ```where``` : recherche
  - ```median```, ```average```, ```std```,```var``` : statistiques classiques
  - ```cov``` : matrice covariante
  - ```histogram``` : histogram
  
scipy :
- Plein de constantes
- **Plein** de fonctions utiles (survol)
  - ```fft```, ```dct```, ```dst``` : fourrier
  - ```quad```, ```simps```, ```odr``` : integration
  - ```solve_ivp```, ```solve_bvp``` : ODE solver
  - ```griddata```, ```make_interp_spline``` : interpolation
  - ```solve```, ```inv```, ```det```, ```eigvals``` : linalg variant
  - ```lu```, ```svd``` : more linalg
  - ```convolve```, ```correlate```, ```gaussian_filter```,```spline_filter``` : filtres
  - ```binary_closing```, ```binary_dilatation```, ```binary_erosion``` : morphologie
  - ```minimize```, ```leastsq```, ```root```, ```fsolve``` : optimisation et recherche de zeros
  - ```find_peaks_cwt```, ```spectrogram``` : traitement du signal
  - ```lu```, ```csr``` : matrices creuses
  - ```bicg```, ```gmres```, ```splu``` : sparse linalg
  - ```shortest_path``` : calcul sur graph
  - ```KDTree```, ```Delaunay```, calcul spatial
  - ```gauss```, ```laplace```, ```uniform```, ```binom``` : distributions de variables aléatoire
  - ```describe```, ```bayes_mvs``` : more statistics
  - ```airy```, ```jv```, ```erf```, ```fresnel``` : fonctions classiques
  
  
  
Matpotlib
---------
- Tout ce qu'il faut pour dessiner en python
  - 1d / 2d / 3d
  - fixe ou animé
  - statique ou interactif (mais plus statique)
- Très proche de la syntaxe matlab (trop ?)
- Très vaste (trop ?)
- Utilisez la gallerie https://matplotlib.org/gallery/index.html

Quelques exemples
-----------------
```
import numpy as np
import matpotlib.pyplot as plt

def f(t):
  return np.exp(-t) * np.cos(2*np.pi*t)

t = np.arange(0.0, 5.0, 0.1)

plt.plot(t1, f(t1))
plt.show()

```

pillow
------
- lire/ecrire des images dans beaucoup de format

Opencv
------
- Traitement d'image

Quelques exemples
-----------------

simpy
-----
Calcul symbolique en python

pandas
------
- largement basé sur numpy
- permet de faire du traitement 'à la excel'

Autre
-----
.... ?
