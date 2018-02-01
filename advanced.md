Avancé
======

Pratique
--------
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
  - enumerate ``` for index, elem in enumerate(list1):```
- utiliser collections.defaultdict(list) pour initialiser un dictionnaire de listes
- utiliser des exceptions
    - try is very fast but except is very slow
    - permet de sortir de plusieurs boucles/fonctions en même temps
    - permet d'être sûr qu'une erreur est traitée (ou explicitement ignorée)
- **never** use a mutable optionnal arg
- use a shallow copy to modify a list in a loop (or be very carefull

```python
from copy import copy
ensemble={"einstein", "albert", "curie"}
for valeur in copy(ensemble):
    if 'bert' in valeur:
        ensemble.discard(valeur)
print(ensemble)
```

- mecanisme d'import
  - Chercher si le module existe.
  - Chercher si le module a déjà été importé. Si oui, s’arrêter ici
  - Si non, chercher si il a été déjà compilé en .pyc.
  - Si ce n’est pas le cas, compiler le fichier .py en .pyc.
- Executer le module 
- se tenir informer de l'évolution du langage (PEP)
- utilisation de décorateur
  - deboggage, timing, ...

```python
from functools import wraps

def PrintAppel(f):
    def avant_f():
        new_f.NbAppels += 1
        print("On rentre dans {}".format(f.__name__))
        
    def apres_f():
        print("On sort de {}".format(f.__name__))
        print("Il s'agissait du {}eme appel".format(new_f.NbAppels))
    
    @wraps(f)
    def new_f(*args, **xargs):
        avant_f()
        res = f(*args, **xargs)
        apres_f()
        return res
    
    new_f.NbAppels = 0
    
    return new_f

```

```python
@PrintAppel
def une_functon(x):
  return 2*x
```

```python
une_functon(2)
```

- Utiliser des classes pour isoler vos operations

```python
class egg(object):                            # Tous les objets dérivent d'object
  """ Full exemple of a class in python """
  total_number = 0                            # attribut partagé par toutes les instance **DANGER** ! 
  
  def __init__(self, number=1):               # constructeur
    """ constructor from number """
    self.number = number                      # Bonne facon de définir les attributs
    egg.total_number += number
    
  @classmethod
  def depuis_recette(cls, ingredients):       # constructeur alternatif (rare)
    """ constructor from recepee """
    return cls(ingredients["oeufs"])
    
  def __del__(self):                          # destructeur (rare)
    """ destructor """
    egg.total_number -= self.number

  def __str__(self):                          # permet d'imprimer l'instance
    """ Permet d'imprimer l'instance """
    return "Sur {} oeufs, j'en possède {}".format(egg.total_number, self.number)
        
  def combien(self):                          # fonction normale de l'instance
    """ Retourne le nombre d'oeufs dans l'instance """
    return self.number

  @staticmethod
  def combien_egg():                          # fonction de l'objet (rare)
    """ Retourne le nombre d'oeufs au total """
    return egg.total_number

oeuf_au_plat = egg()
omelette=egg(3)
recette_crepes={"oeufs":2, "lait":0.5, "farine":300}
crepe = egg.depuis_recette(recette_crepes)
print("Oeuf au plat : ", oeuf_au_plat)
print("Omelette     : ", omelette)
print("Crepes       : ", crepe)

print("{:<12} : {:>5} | {}".format("egg","NaN", egg.combien_egg()))
print("{:<12} : {:>5} | {}".format("oeuf_au_plat",oeuf_au_plat.combien(), oeuf_au_plat.combien_egg()))
print("{:<12} : {:>5} | {}".format("omelette",omelette.combien(), omelette.combien_egg()))
print("{:<12} : {:>5} | {}".format("crepe",crepe.combien(), crepe.combien_egg()))
del omelette
print("{:<12} : {:>5} | {}".format("egg","NaN", egg.combien_egg()))
print("{:<12} : {:>5} | {}".format("oeuf_au_plat",oeuf_au_plat.combien(), oeuf_au_plat.combien_egg()))
print("{:<12} : {:>5} | {}".format("crepe",crepe.combien(), crepe.combien_egg()))
del oeuf_au_plat
del crepe

help(egg)
```

- pour lancer des programes externes, utiliser
  -
```subprocess.check_call(["cmd", "arg1", "arg2"])``` si la sortie du programme
ne vous interesse pas
  - ```data = subprocess.check_output(["cmd", "arg1",
"arg2"])``` sinon (penser a decoder)

```python
import subprocess
data = subprocess.check_output(["ls", "-l", "--color"]).decode('utf-8')
print(data)
```

Packaging
---------
- respecter la PEP (not only for prettyness)
- type hinting (tout nouveau)
  - ca ne change (presque) rien à l'éxecution
  - mypy (et de plus en plus d'IDE) permettent de verifier
  - le module typing permet de définir des types
  - de plus en plus de paquet les definissent

```python
def greeting(name: str) -> str:
    var = "Hello"  # type: str
    # python 3.7 : var = "Hello" : str
    
    return var + " " + name
```

- docstring (auto-documentation)
  - toutes les fonctions
  - toutes les classes
- tous les modules (```__init__.py```)
  - tous les fichiers
- pytest  (unit- testing)
  - auto discovery (use tests folders, test_truc function, and TestMachin classes)
  - allow parametrization

```python
#ONLY for ipython
import ipytest.magics
import pytest
__file__ = 'advanced.ipynb'
```

```python
%%run_pytest[clean] -qq
#this was only for ipython

def test_sorted():
    assert sorted([5, 1, 4, 2, 3]) == [1, 2, 3, 4, 5]
    
# as does parametrize
@pytest.mark.parametrize('input,expected', [
                                            ([2, 1], [1, 2]),
                                            ('zasdqw', list('adqswz')),
                                            ]
                         )
def test_examples(input, expected):
    actual = sorted(input)
    assert actual == expected
```

- gettext (auto-internationnalization) ?
- logging
  - print -> go to console (for ordinary usage)
  - warning.warn -> go to console (usually once : for signaling a something the user should fix)
  - logging.level -> go anywhere you want (for detailled output and/or diagnostic)

```python
import logging
import warnings


def prepare_logging():
    """
    Prepare all logging facilities
    
    This should be done in a separate module
    """

    # if not already done, initialize logging facilities
    logging.basicConfig()

    # create a logger for the current module
    logger = logging.getLogger(__name__)

    ## ONLY FOR IPYTHON
    # clean logger (ipython + multiple call)
    from copy import copy
    for handler in copy(logger.handlers):
        logger.removeHandler(handler)
    # Do not give propagate message to ipython (or else thy will be printed twice)
    logger.propagate=False
    ## ONLY FOR IPYTHON


    # optionnal : change format of the log
    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")

    # optionnal : create a handler for file output
    fileHandler = logging.FileHandler("{logPath}/{fileName}.log".format(logPath=".", fileName="test"))
    # optionnal : create a handler for console output
    consoleHandler = logging.StreamHandler()

    # optionnal : Apply formatter to both handles
    fileHandler.setFormatter(logFormatter)
    consoleHandler.setFormatter(logFormatter)

    # optionnal : attach handler to the logger
    logger.addHandler(fileHandler)
    logger.addHandler(consoleHandler)

    # what severity to log (default is NOTSET, i.e. all)
    logger.setLevel(logging.DEBUG)            # ALL
    fileHandler.setLevel(logging.INFO)        # NO DEBUG
    consoleHandler.setLevel(logging.WARNING)  # ONLY WARNING AND ERRORS

    return logger
    
    
logger = prepare_logging()

def egg():
    warnings.warn("A warning printed once")

egg()
    
logger.info('Start reading database')

records = {'john': 55, 'tom': 66}

logger.debug('Records: {}'.format(records))
logger.info('Updating records ...')
logger.warning("There is only 2 record !")
logger.info('Saving records ...')
logger.error("Something happend, impossible to save the records")
logger.info('Restoring records ...')
logger.critical("Database corrupted !")
logger.info('End of program')

egg()
```


performance
-----------
- profiling : Only optimize the bottlenecks !
  - timeit (for small snippets of code)

```python
%timeit 1+2
%timeit 1*2

```

  - cProfile (for real code)  
   ```python3 -m cProfile -o profile.pstats script.py```  
   ```gprof2dot -f pstats profile.pstats | dot -Tpng -o profile.png```

```python
import numpy as np
import cProfile
import re

def function2(array):
    for i in range(500):
        array += 3
        array = array*2
    return array

def function1():
    array = np.random.randint(500000, size=5000000)
    array = function2(array)
    return sorted(array)

cProfile.run('function1()')
```

- in sequential
    - small is beautifull (PEP 20)
    - inline manually
    - local is faster than global (and avoid dots)
    - choose the right data structure / algorithm
    - prefere numpy based array
    - avoid loops (vectorization using slice)
    - avoid copy of array
    - changing size of an array
    - use the out argument in numpy
- compiler
  - numexpr (only small expression)
  - f2py
    - included with numpy
    - compilation must be done separately
    - be carefull to the memory ordering
  - cython
  - numba

```python
fsource = """
    module themodule
    implicit none
    contains
    
    subroutine fonction1(a,b,c,n)
    implicit none
    integer(kind=8), intent(in) :: n
    double precision,intent(in) :: a(n)
    double precision,intent(in) :: b(n)
    logical,intent(out) :: c(n)

    c = a*b-4.1*a > 2.5*b

    end subroutine fonction1
    
    subroutine fonction2(a,b,c,n)
    implicit none
    integer(kind=8), intent(in) :: n
    double precision,intent(in) :: a(n)
    double precision,intent(in) :: b(n)
    double precision,intent(out) :: c(n)

    c = sin(a) + asinh(a/b)

    end subroutine fonction2
    
    
    subroutine convolve_fortran(f,g,vmax,wmax,smax,tmax,h,err)
    implicit none
    integer(kind=8),intent(in)  :: vmax,wmax,smax,tmax
    integer(kind=8),intent(in)  :: f(vmax,wmax), g(smax,tmax)
    integer(kind=8),intent(out) :: h(vmax,wmax)

    integer(kind=8),intent(out) :: err
    integer(kind=8) :: smid,tmid
    integer(kind=8) :: value
    integer(kind=8) :: x, y, s, t, v, w
    
    ! f is an image and is indexed by (v, w)
    ! g is a filter kernel and is indexed by (s, t),
    !   it needs odd dimensions
    ! h is the output image and is indexed by (v, w),

    err = 0
    if (modulo(smax,2) /= 1 .or. modulo(tmax,2) /= 1) then
        err = 1
        return
    endif
        
    ! smid and tmid are number of pixels between the center pixel
    ! and the edge, ie for a 5x5 filter they will be 2.  
    smid = smax / 2
    tmid = tmax / 2
    
    h = 0
    ! Do convolution
    ! warning : memory layout is different in fortran
    do y=tmid,wmax-tmid-1
        do x=smid,vmax-smid-1
            ! Calculate pixel value for h at (x,y). Sum one component
            ! for each pixel (s, t) of the filter g.
            
            value = 0
            do t=0,tmax - 1
                do s=0,smax - 1
                    v = x - smid + s
                    w = y - tmid + t
                    
                    ! warning : array start at 1 in fortran
                    value = value + g(s + 1, t + 1) * f(v+1, w+1)
                enddo
            enddo
            ! warning : array start at 1 in fortran
            h(x+1, y+1) = value
        enddo
    enddo
    return
    end subroutine convolve_fortran
    end module themodule
    """

with open("fortranModule.f90",'w') as f:
    for line in fsource:
        f.write(line)

import subprocess
try:
    data = subprocess.check_output(["venv/bin/f2py",
                                    "-c", "fortranModule.f90",
                                    "-m", "monModuleFortran",
                                    "--opt='-Ofast -march=native'", "--noarch",
#                                    "--debug-capi", "--debug",
                                    "-DF2PY_REPORT_ON_ARRAY_COPY=1"
                                   ]).decode('utf-8')
except subprocess.CalledProcessError as e:
    print(e.output.decode('utf-8'))
else:
    #print(data)
    print("compilation OK")   
```

```python
def fonction1(a,b):
    return a*b-4.1*a > 2.5*b

def fonction2(a,b):
    return np.sin(a) + np.arcsinh(a/b)


def convolve_python(f, g):
    # f is an image and is indexed by (v, w)
    # g is a filter kernel and is indexed by (s, t),
    #   it needs odd dimensions
    # h is the output image and is indexed by (x, y),

    if g.shape[0] % 2 != 1 or g.shape[1] % 2 != 1:
        raise ValueError("Only odd dimensions on filter supported")
        
    # smid and tmid are number of pixels between the center pixel
    # and the edge, ie for a 5x5 filter they will be 2.
    vmax = f.shape[0]
    wmax = f.shape[1]
    smax = g.shape[0]
    tmax = g.shape[1]
    
    smid = smax // 2
    tmid = tmax // 2

    # Allocate result image.
    h = np.zeros_like(f)
    
    # Do convolution
    for x in range(smid,vmax-smid):
        for y in range(tmid,wmax-tmid):
            # Calculate pixel value for h at (x,y). Sum one component
            # for each pixel (s, t) of the filter g.
            value = 0
            for s in range(smax):
                for t in range(tmax):
                    v = x - smid + s
                    w = y - tmid + t
                    value += g[s, t] * f[v, w]
            h[x, y] = value
    return h
```

```python
%load_ext Cython
```

```python
%%cython
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sin
from libc.math cimport asinh

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def cfonction1(np.ndarray[double, ndim=1] a, np.ndarray[double, ndim=1] b):
    cdef long n = a.shape[0]
    cdef long m = b.shape[0]
    
    if n != m :
        raise ValueError("Arrays must have the same dimension")
        
    cdef long[:] res = np.empty([n], dtype=long)
    
    cdef long i
    for i in range(n):
        res[i] = a[i]*b[i]-4.1*a[i] > 2.5*b[i]
    return res

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def cfonction2(np.ndarray[double, ndim=1] a, np.ndarray[double, ndim=1] b):
    cdef long n = a.shape[0]
    cdef long m = b.shape[0]
    
    if n != m :
        raise ValueError("Arrays must have the same dimension")
        
    cdef double[:] res = np.empty([n], dtype=np.double)
    
    cdef long i
    for i in range(n):
        res[i] = sin(a[i]) + asinh(a[i]/b[i])
    return res


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def convolve_cython(np.ndarray[long, ndim=2] f, np.ndarray[long, ndim=2] g):
    # f is an image and is indexed by (v, w)
    # g is a filter kernel and is indexed by (s, t),
    #   it needs odd dimensions
    # h is the output image and is indexed by (x, y),
    if g.shape[0] % 2 != 1 or g.shape[1] % 2 != 1:
        raise ValueError("Only odd dimensions on filter supported")

    # smid and tmid are number of pixels between the center pixel
    # and the edge, ie for a 5x5 filter they will be 2.
    
    cdef long vmax = f.shape[0]
    cdef long wmax = f.shape[1]
    cdef long smax = g.shape[0]
    cdef long tmax = g.shape[1]
    cdef long smid = smax // 2
    cdef long tmid = tmax // 2
    
    # Allocate result image.
    cdef np.ndarray[long, ndim=2] h = np.zeros([vmax, wmax], dtype=long)

    cdef long value
    cdef long x, y, s, t, v, w

    # Do convolution
    for x in range(smid,vmax-smid):
        for y in range(tmid,wmax-tmid):
            # Calculate pixel value for h at (x,y). Sum one component
            # for each pixel (s, t) of the filter g.
            value = 0
            for s in range(smax):
                for t in range(tmax):
                    v = x - smid + s
                    w = y - tmid + t
                    value += g[s, t] * f[v, w]
            h[x, y] = value
    return h
```

```python
from monModuleFortran import themodule

def convolve_fortran(f,g):
    h, err = themodule.convolve_fortran(f,g)
    if err:
        print(err)
        raise ValueError("FORTRAN ERROR ! (Probably : Only odd dimensions on filter supported)")
    return h
```

```python
import numba as nb


@nb.jit(nopython=True, nogil=True, cache=False, parallel=True)
def nb_fonction1(a, b):
    return a*b-4.1*a > 2.5*b

@nb.jit(nopython=True, nogil=True, cache=False, parallel=True)
def nb_fonction2(a, b):
    return np.sin(a) + np.arcsinh(a/b)

@nb.stencil(standard_indexing=("g",),neighborhood=((-4, 4),(-4, 4)))
def convolve_kernel(f, g):

    smax = g.shape[0]
    tmax = g.shape[1]

    smid = smax // 2
    tmid = tmax // 2
    
    h = 0
    for s in range(smax):
        for t in range(tmax):
            h += g[s, t] * f[s-smid, t-tmid]   
    return h

@nb.jit(nopython=True, nogil=True, cache=False, parallel=True)
def convolve_numba(f, g):

    if g.shape[0] % 2 != 1 or g.shape[1] % 2 != 1:
        raise ValueError("Only odd dimensions on filter supported")

    return convolve_kernel(f,g)

#apply it for compilation
N = 10
f = np.arange(N*N, dtype=np.int).reshape((N,N))
g = np.arange(81, dtype=np.int).reshape((9, 9))
d = convolve_numba(f, g)
a = np.arange(1,5)
b = np.arange(1,5)
nb_fonction1(a, b)
nb_fonction2(a, b)
print("compilation OK")
```

```python
import numpy as np
import numexpr as ne
from monModuleFortran import themodule
```

```python
print("Easy cases")
a = np.arange(1,1e6)   # Choose large arrays for better speedups
b = np.arange(1,1e6)

print("With numpy")
%timeit fonction1(a, b)
%timeit fonction2(a, b)

print("With numexpr")
%timeit ne.evaluate('a*b-4.1*a > 2.5*b')  
%timeit ne.evaluate("sin(a) + arcsinh(a/b)") # numexpr is multithreaded

print("With fortran")
%timeit  c = themodule.fonction1(a, b)
%timeit  d = themodule.fonction2(a, b)

print("With cython")
%timeit  cfonction1(a, b)
%timeit  cfonction2(a, b)

print("With numba")
%timeit  nb_fonction1(a, b)
%timeit  nb_fonction2(a, b)
```

```python
import numpy as np
print("Convolution")
N = 200
f = np.arange(N*N, dtype=np.int).reshape((N,N))
g = np.arange(81, dtype=np.int).reshape((9, 9))
ft = np.asfortranarray(f)                   # memory ordering for fortran
gt = np.asfortranarray(g)

print("With numpy")
%timeit convolve_python(f, g)
a = convolve_python(f, g)

print("With fortran")
%timeit convolve_fortran(ft, gt)
b = convolve_fortran(ft, gt)
print(np.allclose(a,b))

print("With cython")
%timeit convolve_cython(f, g)
c = convolve_cython(f, g)
print(np.allclose(a,c))

print("With numba")
%timeit convolve_numba(f, g)
d = convolve_numba(f, g)
print(np.allclose(a,d))
```

```python
print("Convolution2")
N = 2000
f = np.arange(N*N, dtype=np.int).reshape((N,N))
g = np.arange(81, dtype=np.int).reshape((9, 9))
ft = np.asfortranarray(f)                   # memory ordering for fortran
gt = np.asfortranarray(g)

print("With fortran")
%timeit convolve_fortran(ft, gt)
a = convolve_fortran(ft, gt)
#print(a)

print("With cython")
%timeit convolve_cython(f, g)
b = convolve_cython(f, g)
print(np.allclose(a,b))
#print(b)

print("With numba")
%timeit convolve_numba(f, g)
c = convolve_numba(f, g)
print(np.allclose(a,c))
```

- parallelism
  - cuda

```python
from string import Template

cuda_src_template = Template("""
// Cuda splitting
#define MTB ${max_threads_per_block}
#define MBP ${max_blocks_per_grid}

// Array size
#define fx ${fx}
#define fy ${fy}
#define gx ${gx}
#define gy ${gy}

// Macro for converting subscripts to linear index:
#define f_INDEX(i, j) (i)*(fy)+(j)

// Macro for converting subscripts to linear index:
#define g_INDEX(i, j) (i)*(gy)+(j)

__global__ void convolve_cuda(long *f, long *g, long *h) {

    unsigned int idx = blockIdx.y*MTB*MBP + blockIdx.x*MTB+threadIdx.x;

    // Convert the linear index to subscripts:
    unsigned int i = idx/fy;
    unsigned int j = idx%fy;

    long smax = gx;
    long tmax = gy;

    long smid = smax / 2;
    long tmid = tmax / 2;

    if (smid <= i && i < fx-smid) {
    if (tmid <= j && j < fy-tmid) {

        h[f_INDEX(i,j)] = 0.;
        
        for (long s=0;s<smax;s++)
            for (long t=0;t<tmax;t++)
                h[f_INDEX(i,j)] += g[g_INDEX(s, t)] * f[f_INDEX(i+s-smid, j+t-tmid)];
    
    }
    }
}
""")
```

```python
import skcuda.misc as misc
import pycuda.autoinit
device = pycuda.autoinit.device
max_threads_per_block, _, max_grid_dim = misc.get_dev_attrs(device)
max_blocks_per_grid = max(max_grid_dim)
```

```python
from functools import partial
from pycuda.compiler import SourceModule

cuda_src = cuda_src_template.substitute(max_threads_per_block=max_threads_per_block,
                                        max_blocks_per_grid=max_blocks_per_grid,
                                        fx=f.shape[0], fy=f.shape[1],
                                        gx=g.shape[0], gy=g.shape[1]
                                       )
cuda_module = SourceModule(cuda_src, options= ["-O3", "-use_fast_math", "-default-stream=per-thread"])
print("Compilation OK")

__convolve_cuda = cuda_module.get_function('convolve_cuda')

block_dim, grid_dim = misc.select_block_grid_sizes(device, f.shape)
_convolve_cuda = partial(__convolve_cuda,
                         block=block_dim,
                         grid=grid_dim)
```

```python
import pycuda.gpuarray as gpuarray

f_gpu = gpuarray.to_gpu(f)
g_gpu = gpuarray.to_gpu(g)

def convolve_cuda(f_gpu,g_gpu):
    h_gpu = gpuarray.zeros_like(f_gpu)
    _convolve_cuda(f_gpu, g_gpu, h_gpu)
    return h_gpu.get()
```

```python
%timeit convolve_cuda(f_gpu, g_gpu)
d = convolve_cuda(f_gpu, g_gpu)
print(np.allclose(a,d))
```

- asyncio
    - not a real parallelism
    - effective for io-bound tasks (web)
- not very interesting here
- multithreading
    - more parallelism (GIL)
    - shared memory
- multiprocessing
    - real parallelism
    - limited to one
computer
    - two main implementation
        - multiprocessing (stdlib) which
is flexible
        - joblib which is relatively easy to use
- mpi (mpi4py)
- real parallelism
    - unlimited
    - relatively complex to use (same as in
C, fortran, ...)

```python
import time
import numpy as np
def heavy_fonction(i):
    t = np.random.rand()/10
    time.sleep(t)
    return i,t
    
```

```python
from math import sqrt
from joblib import Parallel, delayed
tic = time.time()
res = Parallel(n_jobs=-1)(delayed(heavy_fonction)(i) \
                            for i in range(2000))
tac = time.time()
index, times = np.asarray(res).T
print(tac - tic)
print(times.sum())
```

```python
from threading import Thread, RLock

N = 2000
N_t = 10
input_list = np.arange(N,dtype=int)
current = 0
nprocs = 8
output_list = np.empty(N)

verrou = RLock()

class ThreadJob(Thread):
    def run(self):
        global current
        """Code à exécuter pendant l'exécution du thread."""
        while current < N:
            
            with verrou:
                position = current
                current += N_t
            
            fin = min(position+N_t+1,N)
            
            for i in range(position, fin):
                j,t = heavy_fonction(input_list[i])
                output_list[j] = t

# Création des threads
threads = [ThreadJob() for i in range(nprocs)]

tic = time.time()
# Lancement des threads
for thread in threads:
    thread.start()
    
# Attend que les threads se terminent
for thread in threads:
    thread.join()
tac = time.time()


print(tac - tic)
print(output_list.sum())
```

```python
import multiprocessing as mp
from queue import Empty

def process_job(q,r):
    while True:
        try:
            i = q.get(block=False)
            r.put(heavy_fonction(i))
        except Empty:
            if q.empty():
                if q.qsize() == 0:
                    break


# Define an output queue
r = mp.Queue()

# Define an input queue
q = mp.Queue()

for i in range(2000):
    q.put(i)

nprocs = 8
# Setup a list of processes that we want to run
processes = [mp.Process(target=process_job, args=(q, r)) for i in range(nprocs)]

tic = time.time()

# Run processes
for p in processes:
    p.start()
    
# Get process results from the output queue
results = np.empty(2000)
for i in range(2000):
    j,t = r.get()
    results[j] = t
    
tac = time.time()

# Exit the completed processes
for p in processes:
    p.join()
    
print(tac - tic)
print(results.sum())
```
