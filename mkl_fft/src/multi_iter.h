/*
 Copyright (c) 2017, Intel Corporation

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

     * Redistributions of source code must retain the above copyright notice,
       this list of conditions and the following disclaimer.
     * Redistributions in binary form must reproduce the above copyright
       notice, this list of conditions and the following disclaimer in the
       documentation and/or other materials provided with the distribution.
     * Neither the name of Intel Corporation nor the names of its contributors
       may be used to endorse or promote products derived from this software
       without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
 FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef MULTI_ITER_H
#define MULTI_ITER_H

#include "numpy/arrayobject.h"

typedef struct multi_iter_t {
  npy_intp *shape; // shape of the tensor
  npy_intp *ind;   // multi-index
  int  rank;   // tensor rank, length of shape and of ind
  char done;   // boolean variable: True when end of iterator has been reached
} multi_iter_t;

typedef struct multi_iter_masked_t {
  npy_intp *shape; // shape of the tensor
  npy_intp *ind;   // multi-index
  int  *mask;  // list of indexes to iterate over
  int  rank;   // tensor rank, length of shape and of ind
  int  mask_len;
  char done;   // boolean variable: True when end of iterator has been reached
} multi_iter_masked_t;


#define MultiIter_Index(mit) ((mit)->ind)
#define MultiIter_Shape(mit) ((mit)->shape)
#define MultiIter_Done(mit)  ((mit)->done)
#define MultiIter_Rank(mit)  ((mit)->rank)
#define MultiIter_Mask(mit)  ((mit)->mask)
#define MultiIter_MaskLength(mit)  ((mit)->mask_len)

#define MultiIter_IndexElem(mit, i) ((mit)->ind)[(i)]
#define MultiIter_ShapeElem(mit, i) ((mit)->shape)[(i)]
#define MultiIter_MaskElem(mit, i) ((mit)->mask)[(i)]

multi_iter_t* multi_iter_new(npy_intp*, int);
multi_iter_masked_t* multi_iter_masked_new(npy_intp*, int, int*, int);

void multi_iter_free(multi_iter_t*);
int multi_iter_next(multi_iter_t*);

void multi_iter_masked_free(multi_iter_masked_t*);
int multi_iter_masked_next(multi_iter_masked_t*);


#endif
