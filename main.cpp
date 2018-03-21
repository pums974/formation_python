#include <vector>
#include <cmath> 
#include <string>
#include <iostream>
#include <omp.h>

#include <ctime>
#include <sys/time.h>

double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}
double get_cpu_time(){
    return (double)clock() / CLOCKS_PER_SEC;
}


using namespace std;

vector<bool> function1(const vector<double> &a,
                       const vector<double> &b)
{
    const size_t N = a.size();

    vector<bool> res(N);
    
    #pragma omp parallel for  num_threads(8)
    for (size_t i=0; i<N; i++){
        res[i] = a[i] * b[i] - 4.1 * a[i] > 2.5 * b[i];
    } 
    return res;
}


vector<double> function2(const vector<double> &a,
                         const vector<double> &b)
{
    const size_t N = a.size();

    vector<double> res(N);
    
    #pragma omp parallel for  num_threads(8)
    for (size_t i=0; i<N; i++){
        res[i] = sin(a[i]) + asinh(a[i] / b[i]);
    } 
    return res;
}


void convolve(const vector< vector<long> > &f,
              const vector< vector<long> > &g,
              vector< vector<long> > &h)
{
    /*
     * f is an image and is indexed by (v, w)
     * g is a filter kernel and is indexed by (s, t),
     * it needs odd dimensions
     * h is the output image and is indexed by (x, y),
    */
    
    const size_t vmax = f.size();
    const size_t wmax = f[0].size();
    const size_t smax = g.size();
    const size_t tmax = g[0].size();
    
    if (smax % 2 == 0 || tmax % 2 == 0)
    {
        throw string("Only odd dimensions on filter supported");
    }
    
    // smid and tmid are number of pixels between the center pixel
    // and the edge, ie for a 5x5 filter they will be 2.
    
    const size_t smid = smax / 2;
    const size_t tmid = tmax / 2;
    
    // Allocate result image.
    for(size_t x = 0; x < vmax; ++x) {
        for(size_t y = 0; y < wmax; ++y) {
            h[x][y] = 0;
        }
    }


    // Do convolution
    #pragma omp parallel for  num_threads(8)
    for (size_t x=smid; x<vmax - smid; x++)
    {
        for (size_t y=tmid; y<wmax - tmid; y++)
        {
            /*
             * Calculate pixel value for h at (x,y). Sum one component
             * for each pixel (s, t) of the filter g.
            */

            long value = 0;
            for (size_t s=0; s<smax; s++)
            {
                for (size_t t=0; t<tmax; t++)
                {
                    size_t v = x - smid + s;
                    size_t w = y - tmid + t;
                    value += g[s][t] * f[v][w];
                }
            }
            h[x][y] = value;
        }
    }           
}



vector<double> data1D(const size_t N)
{
    vector<double> data(N);

    for (size_t i=0; i<N; i++){
        data[i] = double(i+1);
    }
    
    return data;
}

vector< vector<long> >  data2D(const size_t N, const size_t M)
{
    vector< vector<long> > data(N, vector<long>(M));

    for (size_t i=0; i<N; i++){
        for (size_t j=0; j<M; j++){
            data[i][j] = double(i * M + j);
        }
    }
    
    return data;
}


int main(int argc, char *argv[])
{

    vector<double> a;
    vector<double> b;
    vector< vector<long> > f;
    vector< vector<long> > g;

    a =  data1D(1000000);
    b =  data1D(1000000);

    cout << "Warm up the CPU" << endl;
    double start = get_wall_time();
    for (size_t i=0; (get_wall_time() - start) < 10.; i++)
    {
        vector<bool> c = function1(a,b);
    }

    cout << "Testing function1" << endl;
    start = get_wall_time();
    for (size_t i=0; i< 700; i++)
    {
        vector<bool> c = function1(a,b);
    }
    cout << "Time: " << (get_wall_time() - start) / double(700. / 1000.) << " ms" << endl;

    cout << "Testing function2" << endl;
    start = get_wall_time();
    for (size_t i=0; i< 70; i++)
    {
        vector<double> c = function2(a,b);
    }
    cout << "Time: " << (get_wall_time() - start) / double(70. / 1000.) << " ms" << endl;

    f =  data2D(200, 200);
    g =  data2D(9, 9);
    vector< vector<long> > h(200, vector<long>(200));

    cout << "Testing convolution small case" << endl;
    start = get_wall_time();
    for (size_t i=0; i< 1000; i++)
    {
        convolve(f,g,h);
    }
    cout << "Time: " << (get_wall_time() - start) / double(1000. / 1000.) << " ms" << endl;

    f =  data2D(2000, 2000);
    h = vector< vector<long> > (2000, vector<long>(2000));

    cout << "Testing convolution large case" << endl;
    start = get_wall_time();
    for (size_t i=0; i< 70; i++)
    {
        convolve(f,g,h);
    }
    cout << "Time: " << (get_wall_time() - start) / double(70. / 1000.) << " ms" << endl;
}
