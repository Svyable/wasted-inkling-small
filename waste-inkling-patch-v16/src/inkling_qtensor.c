/* SPDX-License-Identifier: Apache-2.0 */
#define _POSIX_C_SOURCE 200809L
#include "inkling_qtensor.h"
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <io.h>
#include <sys/stat.h>
#include <windows.h>
#define OPEN_RO(p) _open((p), _O_RDONLY | _O_BINARY)
#define CLOSE_FD(fd) _close(fd)
#else
#include <sys/stat.h>
#include <unistd.h>
#define OPEN_RO(p) open((p), O_RDONLY)
#define CLOSE_FD(fd) close(fd)
#endif
#define IKQT_MAGIC 0x54514b49u
#define IKQT_VERSION 1u
#define IKQT_HEADER 96u
#define IKQT_Q8 2
#define IKQT_Q4 3

static uint16_t rd16(const unsigned char *p){return (uint16_t)(p[0]|((uint16_t)p[1]<<8));}
static uint32_t rd32(const unsigned char *p){return (uint32_t)p[0]|((uint32_t)p[1]<<8)|((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24);}
static uint64_t rd64(const unsigned char *p){return (uint64_t)rd32(p)|((uint64_t)rd32(p+4)<<32);}
static uint32_t crc32_update(uint32_t c,const unsigned char*p,size_t n){for(size_t i=0;i<n;i++){c^=p[i];for(int k=0;k<8;k++)c=(c>>1)^((0u-(c&1u))&0xedb88320u);}return c;}
static uint32_t name_crc(const char*s){return ~crc32_update(~0u,(const unsigned char*)s,strlen(s));}
static float f16(uint16_t h){unsigned s=h>>15,e=(h>>10)&31u,m=h&1023u;union{uint32_t u;float f;}v;if(!e){if(!m)v.u=s<<31;else{int ee=1;while(!(m&1024u)){m<<=1;ee--;}m&=1023u;v.u=(s<<31)|((uint32_t)(ee+112)<<23)|(m<<13);}}else if(e==31)v.u=(s<<31)|0x7f800000u|(m<<13);else v.u=(s<<31)|((e+112u)<<23)|(m<<13);return v.f;}
static int preadn(int fd,void*buf,size_t n,uint64_t off){unsigned char*p=buf;
#ifdef _WIN32
HANDLE h=(HANDLE)_get_osfhandle(fd);if(h==INVALID_HANDLE_VALUE)return -1;while(n){DWORD ask=n>0x7fffffffu?0x7fffffffu:(DWORD)n,got=0;OVERLAPPED ov;memset(&ov,0,sizeof ov);ov.Offset=(DWORD)off;ov.OffsetHigh=(DWORD)(off>>32);if(!ReadFile(h,p,ask,&got,&ov)||!got)return -1;p+=got;n-=got;off+=got;}
#else
while(n){ssize_t r=pread(fd,p,n,(off_t)off);if(r<=0)return -1;p+=(size_t)r;n-=(size_t)r;off+=(uint64_t)r;}
#endif
return 0;}
static int verify_range(int fd,uint64_t off,uint64_t n,uint32_t want){unsigned char b[65536];uint32_t c=~0u;while(n){size_t z=n>sizeof b?sizeof b:(size_t)n;if(preadn(fd,b,z,off))return -1;c=crc32_update(c,b,z);off+=z;n-=z;}return ~c==want?0:-1;}

int waste_inkling_qtensor_open(waste_inkling_qtensor*t,const char*path,const char*name,const int*shape,int ndim,int verify){unsigned char h[IKQT_HEADER];if(!t||!path||!name||!shape||ndim<1||ndim>4)return -1;memset(t,0,sizeof*t);t->fd=-1;int fd=OPEN_RO(path);if(fd<0||preadn(fd,h,sizeof h,0)){if(fd>=0)CLOSE_FD(fd);return -1;}if(rd32(h)!=IKQT_MAGIC||rd16(h+4)!=IKQT_VERSION||(h[6]!=IKQT_Q8&&h[6]!=IKQT_Q4)||h[7]!=(unsigned)ndim||rd32(h+8)<2||rd32(h+68)!=name_crc(name)){CLOSE_FD(fd);return -1;}for(int i=0;i<4;i++){int got=(int)rd32(h+12+4*i);if(i<ndim){if(got!=shape[i]||got<1){CLOSE_FD(fd);return -1;}}else if(got){CLOSE_FD(fd);return -1;}}
uint64_t rows=rd64(h+28),cols=rd64(h+36),qb=rd64(h+44),sb=rd64(h+52);uint32_t group=rd32(h+8);size_t groups=(size_t)((cols+group-1)/group);size_t rowbytes=groups*(size_t)group*(h[6]==IKQT_Q8?1u:1u)/ (h[6]==IKQT_Q8?1u:2u);if(!rows||!cols||qb!=rows*rowbytes||sb!=rows*groups*2u){CLOSE_FD(fd);return -1;}for(int i=72;i<96;i++)if(h[i]){CLOSE_FD(fd);return -1;}t->qrow=malloc(rowbytes?rowbytes:1);t->scales=malloc(groups*sizeof(uint16_t));if(!t->qrow||!t->scales){waste_inkling_qtensor_close(t);CLOSE_FD(fd);return -1;}t->fd=fd;t->fmt=h[6];t->ndim=ndim;t->group=(int)group;for(int i=0;i<ndim;i++)t->shape[i]=shape[i];t->rows=rows;t->cols=cols;t->qbytes=qb;t->scale_bytes=sb;t->scale_off=IKQT_HEADER+qb;t->rowbytes=rowbytes;t->groups=groups;t->q_crc32=rd32(h+60);t->scale_crc32=rd32(h+64);t->verify_crc=verify;if(verify&&(verify_range(fd,IKQT_HEADER,qb,t->q_crc32)||verify_range(fd,t->scale_off,sb,t->scale_crc32))){waste_inkling_qtensor_close(t);return -1;}return 0;}
void waste_inkling_qtensor_close(waste_inkling_qtensor*t){if(!t)return;if(t->fd>=0)CLOSE_FD(t->fd);free(t->qrow);free(t->scales);free(t->qdata);free(t->scale_data);memset(t,0,sizeof*t);t->fd=-1;}
size_t waste_inkling_qtensor_resident_bytes(const waste_inkling_qtensor *t)
{
    if (!t) return 0;
    if (t->qbytes > SIZE_MAX - t->scale_bytes) return 0;
    return (size_t)(t->qbytes + t->scale_bytes);
}
int waste_inkling_qtensor_load_resident(waste_inkling_qtensor *t)
{
    if (!t || t->fd < 0) return -1;
    if (t->qdata && t->scale_data) return 0;
    if (t->qbytes > SIZE_MAX || t->scale_bytes > SIZE_MAX) return -1;
    unsigned char*q=malloc((size_t)t->qbytes?t->qbytes:1);
    uint16_t*s=malloc((size_t)t->scale_bytes?t->scale_bytes:2);
    if(!q||!s){free(q);free(s);return -1;}
    if(preadn(t->fd,q,(size_t)t->qbytes,IKQT_HEADER)||
       preadn(t->fd,s,(size_t)t->scale_bytes,t->scale_off)){free(q);free(s);return -1;}
    t->qdata=q;t->scale_data=s;return 0;
}
int waste_inkling_qtensor_row(waste_inkling_qtensor *t, uint64_t row,
                              float *out, size_t cols)
{
    if (!t || t->fd < 0 || !out || row >= t->rows ||
        cols != (size_t)t->cols) return -1;
    const unsigned char *qrow;
    const uint16_t *sc;
    if (t->qdata && t->scale_data) {
        qrow = t->qdata + row * t->rowbytes;
        sc = t->scale_data + row * t->groups;
    } else {
        if (preadn(t->fd, t->qrow, t->rowbytes,
                   IKQT_HEADER + row * t->rowbytes) ||
            preadn(t->fd, t->scales, t->groups * 2,
                   t->scale_off + row * t->groups * 2)) return -1;
        qrow = t->qrow;
        sc = t->scales;
    }
    for (size_t g = 0; g < t->groups; g++) {
        const float scale = f16(sc[g]);
        const size_t begin = g * (size_t)t->group;
        size_t stop = begin + (size_t)t->group;
        if (stop > cols) stop = cols;
        for (size_t c = begin; c < stop; c++) {
            int q;
            if (t->fmt == IKQT_Q8) q = (int)((const int8_t *)qrow)[c];
            else {
                const unsigned char b = qrow[c >> 1];
                q = (int)((c & 1) ? (b >> 4) : (b & 15)) - 8;
            }
            out[c] = (float)q * scale;
        }
    }
    return 0;
}

int waste_inkling_qtensor_matvec_rows(waste_inkling_qtensor *t, const float *x,
                                      float *out, size_t row0, size_t rows,
                                      size_t cols)
{
    if (!t || !x || !out || cols != (size_t)t->cols ||
        row0 > (size_t)t->rows || rows > (size_t)t->rows - row0) return -1;
    for (size_t r = 0; r < rows; r++) {
        const size_t rr = row0 + r;
        const unsigned char *qrow;
        const uint16_t *sc;
        if (t->qdata && t->scale_data) {
            qrow = t->qdata + rr * t->rowbytes;
            sc = t->scale_data + rr * t->groups;
        } else {
            if (preadn(t->fd, t->qrow, t->rowbytes,
                       IKQT_HEADER + rr * t->rowbytes) ||
                preadn(t->fd, t->scales, t->groups * 2,
                       t->scale_off + rr * t->groups * 2)) return -1;
            qrow = t->qrow;
            sc = t->scales;
        }
        double sum = 0.0;
        for (size_t g = 0; g < t->groups; g++) {
            const float scale = f16(sc[g]);
            const size_t begin = g * (size_t)t->group;
            size_t stop = begin + (size_t)t->group;
            if (stop > cols) stop = cols;
            for (size_t c = begin; c < stop; c++) {
                int q;
                if (t->fmt == IKQT_Q8) q = (int)((const int8_t *)qrow)[c];
                else {
                    const unsigned char b = qrow[c >> 1];
                    q = (int)((c & 1) ? (b >> 4) : (b & 15)) - 8;
                }
                sum += (double)((float)q * scale) * x[c];
            }
        }
        out[r] = (float)sum;
    }
    return 0;
}
int waste_inkling_qtensor_matvec(waste_inkling_qtensor *t, const float *x,
                                 float *out, size_t rows, size_t cols)
{
    return t && rows == (size_t)t->rows
         ? waste_inkling_qtensor_matvec_rows(t, x, out, 0, rows, cols) : -1;
}
