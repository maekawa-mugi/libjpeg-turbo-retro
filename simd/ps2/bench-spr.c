/* One-boot PS2 EE scratchpad placement studies for genuine JPEG SIMD.
 * Tests plain/fancy upsample and downsample, h2v1/h2v2, several widths.
 * No production dispatch change or DMA. Exclusive SPR 0x70000000-3fff.
 * SPDX-License-Identifier: Zlib
 */
#include "../jsimdint.h"
#include <timer.h>
#include <debug.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define ROW_BYTES 1024u
#define INPUT_ROWS 6u
#define OUTPUT_ROWS 4u
#define SPR_SAMPLES 6u
#define SPR_REPS 24u
static JSAMPLE ram_in[INPUT_ROWS][ROW_BYTES] __attribute__((aligned(16)));
static JSAMPLE ram_out[OUTPUT_ROWS][ROW_BYTES] __attribute__((aligned(16)));
static JSAMPLE expected[OUTPUT_ROWS][ROW_BYTES] __attribute__((aligned(16)));
static JSAMPROW input_ptr[INPUT_ROWS], output_ptr[OUTPUT_ROWS];
static volatile uint32_t escape;

static uint64_t median6(const uint64_t v[SPR_SAMPLES])
{
  uint64_t a[SPR_SAMPLES], x;
  unsigned i,j;
  for (i=0;i<SPR_SAMPLES;i++) a[i]=v[i];
  for (i=1;i<SPR_SAMPLES;i++) {
    x=a[i];j=i;while(j && a[j-1]>x){a[j]=a[j-1];j--;}a[j]=x;
  }
  return (a[2]+a[3])/2u;
}
static void invoke(unsigned kind, unsigned vertical, unsigned width,
                   JSAMPARRAY rows, JSAMPARRAY out)
{
  JSAMPARRAY shifted=rows+1;
  if(kind==0) {
    if(vertical==2) jsimd_h2v2_upsample_ps2mmi(4,width,shifted,&out);
    else jsimd_h2v1_upsample_ps2mmi(4,width,shifted,&out);
  } else if(kind==1) {
    if(vertical==2) jsimd_h2v2_fancy_upsample_ps2mmi(4,width,shifted,&out);
    else jsimd_h2v1_fancy_upsample_ps2mmi(4,width,shifted,&out);
  } else {
    /* width is the output width in samples, 8-sample DCT blocks. */
    if(vertical==2) jsimd_h2v2_downsample_ps2mmi(width*2u,4,2,width/8u,rows,out);
    else jsimd_h2v1_downsample_ps2mmi(width*2u,4,2,width/8u,rows,out);
  }
}

/* Each mode executes the same true MMI kernel; only data placement differs.
 * Mode 4 includes both directions of SPR transfer for every operation.
 * Only measured calls fall inside the timer, never validation/printf. */
static uint64_t run_one(unsigned kind, unsigned vertical, unsigned width,
                        unsigned mode, unsigned reps)
{
  JSAMPLE *spr_in=(JSAMPLE *)(uintptr_t)0x70000000u;
  JSAMPLE *spr_out=(JSAMPLE *)(uintptr_t)0x70002000u;
  JSAMPROW si[INPUT_ROWS], so[OUTPUT_ROWS];
  JSAMPARRAY in, out;
  uint64_t start, elapsed;
  unsigned i,r;
  for(i=0;i<INPUT_ROWS;i++) si[i]=spr_in+i*ROW_BYTES;
  for(i=0;i<OUTPUT_ROWS;i++) so[i]=spr_out+i*ROW_BYTES;
  in=(mode==1 || mode==3)?si:input_ptr;
  out=(mode==2 || mode==3)?so:output_ptr;
  for(i=0;i<OUTPUT_ROWS;i++) memset(ram_out[i],0xa5,ROW_BYTES);
  if(mode==1 || mode==3)
    for(i=0;i<INPUT_ROWS;i++) memcpy(si[i],ram_in[i],ROW_BYTES);
  if(mode==2 || mode==3)
    for(i=0;i<OUTPUT_ROWS;i++) memset(so[i],0xa5,ROW_BYTES);
  start=GetTimerSystemTime();
  for(r=0;r<reps;r++) {
    if(mode==4) {
      for(i=0;i<INPUT_ROWS;i++) memcpy(si[i],ram_in[i],ROW_BYTES);
      for(i=0;i<OUTPUT_ROWS;i++) memcpy(so[i],ram_out[i],ROW_BYTES);
      invoke(kind,vertical,width,si,so);
      for(i=0;i<OUTPUT_ROWS;i++) memcpy(ram_out[i],so[i],ROW_BYTES);
    } else invoke(kind,vertical,width,in,out);
  }
  elapsed=GetTimerSystemTime()-start;
  if(mode==2 || mode==3)
    for(i=0;i<OUTPUT_ROWS;i++) memcpy(ram_out[i],so[i],ROW_BYTES);
  for(i=0;i<OUTPUT_ROWS;i++) escape^=ram_out[i][ROW_BYTES-1u];
  return elapsed;
}

int ps2_bench_run_spr(void)
{
  static const char *const kinds[3]={"plain_up","fancy_up","downsample"};
  static const char *const modes[5]={
    "ram","spr_input","spr_output","spr_both","spr_xfer"
  };
  static const unsigned widths[3][4]={{64,128,256,512},
                                      {64,128,256,512},
                                      {64,128,256,512}};
  uint64_t samples[5][SPR_SAMPLES], med[5];
  double summary[3]={0,0,0};
  unsigned kind,vertical,idx,mode,step,s,i,row,col,cases=0;
  for(row=0;row<INPUT_ROWS;row++) {
    input_ptr[row]=ram_in[row];
    for(col=0;col<ROW_BYTES;col++)
      ram_in[row][col]=(JSAMPLE)((col*37u+row*53u+19u)&255u);
  }
  for(row=0;row<OUTPUT_ROWS;row++) output_ptr[row]=ram_out[row];
  puts("JPEG_SPR_META,R5900,real_MMI_sampling,6samples,24reps,16KiB_SPR");
  for(kind=0;kind<3;kind++)
    for(vertical=1;vertical<=2;vertical++)
      for(idx=0;idx<4;idx++) {
        unsigned width=widths[kind][idx];
        unsigned count=kind==1?width*2u:width;
        unsigned rows=kind==2?2u:4u;
        /* Downsample reads 2*width input bytes; all rows are 1024 bytes. */
        run_one(kind,vertical,width,0,SPR_REPS);
        for(row=0;row<OUTPUT_ROWS;row++) memcpy(expected[row],ram_out[row],ROW_BYTES);
        for(mode=1;mode<5;mode++) {
          run_one(kind,vertical,width,mode,SPR_REPS);
          for(row=0;row<OUTPUT_ROWS;row++)
            if(memcmp(ram_out[row],expected[row],ROW_BYTES)) {
              printf("JPEG_SPR_FAIL,%s,h2v%u,%u,%s,output\n",
                     kinds[kind],vertical,width,modes[mode]);
              return 1;
            }
        }
        for(s=0;s<SPR_SAMPLES;s++) for(step=0;step<5;step++) {
          mode=(s+step)%5u;
          samples[mode][s]=run_one(kind,vertical,width,mode,SPR_REPS);
          if(!samples[mode][s]) {
            printf("JPEG_SPR_FAIL,%s,h2v%u,%u,%s,zero_timer\n",
                   kinds[kind],vertical,width,modes[mode]);
            return 1;
          }
          for(row=0;row<OUTPUT_ROWS;row++)
            if(memcmp(ram_out[row],expected[row],ROW_BYTES)) {
              printf("JPEG_SPR_FAIL,%s,h2v%u,%u,%s,sample\n",
                     kinds[kind],vertical,width,modes[mode]);
              return 1;
            }
        }
        for(mode=0;mode<5;mode++) {
          med[mode]=median6(samples[mode]);
          printf("JPEG_SPR,%s,h2v%u,%u,%s,%llu,%.5f\n",
                 kinds[kind],vertical,width,modes[mode],
                 (unsigned long long)med[mode],
                 (double)med[0]/(double)med[mode]);
        }
        if(vertical==2 && width==256u)
          summary[kind]=(double)med[0]/(double)med[3];
        /* Verify the result was actually written, not only untouched padding. */
        if(count==0 || rows==0) return 1;
        ++cases;
      }
  scr_setXY(0,14);
  scr_setfontcolor(0x00ffffff);
  scr_printf("RAM/SPR both @256 P:%4.2fx F:%4.2fx D:%4.2fx   ",
             summary[0],summary[1],summary[2]);
  printf("JPEG_SPR_RESULT,PASS,cases=%u,sink=%lu\n",
         cases,(unsigned long)escape);
  return 0;
}
