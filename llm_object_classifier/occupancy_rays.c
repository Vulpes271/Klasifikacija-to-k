/* One inverse sensor-model update per cell per scan; occupied wins over free.
 * Coordinates are bottom-up grid indices. Removed/invalid beams do NOTHING.
 * No endpoint or free-space evidence can leak behind an omitted hit. */
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
void integrate(float *grid, uint8_t *known, int w, int h,
               const int32_t *origins, const int32_t *ends,
               const uint8_t *valid, int n, float hit, float free_log, float limit) {
    uint8_t *evidence=calloc((size_t)w*h,1);
    for (int i=0;i<n;i++) {
        if (!valid[i]) continue;
        int x=origins[2*i],y=origins[2*i+1],ex=ends[2*i],ey=ends[2*i+1];
        /* Origin must lie in the declared grid; endpoints may be outside. */
        if (x<0||x>=w||y<0||y>=h) continue;
        int dx=abs(ex-x),sx=x<ex?1:-1,dy=-abs(ey-y),sy=y<ey?1:-1,err=dx+dy;
        for (;;) {
            if (x<0||x>=w||y<0||y>=h) break;
            int k=y*w+x;
            if (x==ex && y==ey) {evidence[k]=2;break;}
            if (!evidence[k]) evidence[k]=1;
            int e2=2*err;
            if (e2>=dy) {err+=dy;x+=sx;}
            if (e2<=dx) {err+=dx;y+=sy;}
        }
    }
    for (int k=0;k<w*h;k++) if (evidence[k]) {
        known[k]=1;grid[k]=fmaxf(-limit,fminf(limit,grid[k]+(evidence[k]==2?hit:free_log)));
    }
    free(evidence);
}
