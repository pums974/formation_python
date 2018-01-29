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

scikits
------
'Plugins' for scipy
http://scikits.appspot.com/scikits
e.g. scikit-image

pillow
------
- lire/ecrire des images dans beaucoup de format

Opencv
------
- Traitement d'image


Quelques exemples
-----------------
```
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import data
from skimage.color import label2rgb

# get data
coins = data.coins()

def plot_result(segmentation, title = ""):
    labeled_coins, _ = ndi.label(segmentation)
    image_label_overlay = label2rgb(labeled_coins, image=coins)
    
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(coins, cmap=plt.cm.gray)
    axes[0].contour(segmentation, [0.5], linewidths=1.2, colors='y')
    axes[1].imshow(image_label_overlay)
    if title:
        plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def segmentation_with_threshold(datain, threshold):
    """ simple segmentation based on a threshold """
    return ndi.binary_fill_holes(datain>threshold)


# plot coins
fig, axes = plt.subplots(1,2)
axes[0].imshow(coins, cmap='gray')
axes[1].hist(coins.flatten(), bins=np.arange(0, 256))
plt.show()

# simple extraction based on threshold
for threshold in [100,140,110]:
    segmentation = segmentation_with_threshold(coins, threshold)
    plot_result(segmentation, "Threshold method with {}".format(threshold))
```
```
def segmentation_with_edges(datain):
    """ segmentation based on edges """
    from skimage.feature import canny
    return ndi.binary_fill_holes(canny(datain/255.))
```
```
def filter_small_objects(datain, threshold):
    """ segmentation based on edges """
    # label each zone
    label_objects, nb_labels = ndi.label(datain)
    # size of each zone in pixel
    sizes = np.bincount(label_objects.ravel())
    # mask on all zone bigger than 20
    mask_sizes = sizes > threshold
    # remove the background
    mask_sizes[0] = False
      
    #apply filter
    return mask_sizes[label_objects]  
```
```
def segmentation_with_region(datain):
    """ segmentation based on region """
    from skimage.filters import sobel
    from skimage.morphology import watershed


    elevation_map = sobel(datain)
    
    markers = np.zeros_like(datain)
    # this is background for sure
    markers[datain < 30] = 1
    # this is coin for sure
    markers[datain > 150] = 2
    
    # apply watershed algo
    segmentation = watershed(elevation_map, markers) - 1
    return ndi.binary_fill_holes(segmentation)          
```

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
