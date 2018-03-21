#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <CModule.h>

int64_t timespecDiff(struct timespec *timeA_p, struct timespec *timeB_p)
{
  return ((timeA_p->tv_sec * 1000000000) + timeA_p->tv_nsec) -
           ((timeB_p->tv_sec * 1000000000) + timeB_p->tv_nsec);
}

int main() {
  const size_t vmax = 2000;
  const size_t wmax = 2000;
  const size_t smax = 9;
  const size_t tmax = 9;
  
  long* f = malloc(vmax*wmax*sizeof(long));
  long* h = malloc(vmax*wmax*sizeof(long));
  long* g = malloc(smax*tmax*sizeof(long));
  
  for(size_t ii = 0; ii<vmax*wmax; ii++)
    {
      f[ii] = ii;
    }

  for(size_t ii = 0; ii<smax*tmax; ii++)
    {
      g[ii] = ii;
    }
  
  struct timespec start;
  struct timespec end;
  clock_gettime(CLOCK_REALTIME, &start);
  for (int i=0;i<70;i++)
    {
      convolve_c(f,g,h,vmax,wmax,smax,tmax);
    }
  clock_gettime(CLOCK_REALTIME, &end);

  int64_t timeElapsed = timespecDiff(&end, &start)/70;
  int s = timeElapsed/1000000000;
  int ms = timeElapsed/1000000 - 1000 *s;
  int us = timeElapsed/1000 - 1000000 *s - 1000 * ms;
  int ns = timeElapsed - 1000000000 *s - 1000000 * ms - 1000 * us;
  printf("Time : %4ds %4dms %4dus %4dns\n", s, ms, us, ns);

  free(f);
  free(g);
  free(h);

}
