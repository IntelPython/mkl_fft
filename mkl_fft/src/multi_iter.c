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

#include "multi_iter.h"
#include "mkl.h"
#include <string.h>
#include <assert.h>

multi_iter_t* multi_iter_new(npy_intp shape[], int rank) {
    multi_iter_t *mi;
    int i;
    char d = 0;

    assert(rank > 0);

    mi = (multi_iter_t *) mkl_malloc(sizeof(multi_iter_t), 64);
    MultiIter_Index(mi) = (npy_intp *) mkl_calloc(rank,  sizeof(npy_intp), 64);
    MultiIter_Shape(mi) = (npy_intp *) mkl_malloc(rank * sizeof(npy_intp), 64);
    memcpy(MultiIter_Shape(mi), shape, rank * sizeof(npy_intp));
    MultiIter_Rank(mi) = rank;

    for(i=0; i < rank; i++) {
	d |= MultiIter_IndexElem(mi, i) >= MultiIter_ShapeElem(mi, i);
	if (d) break;
    }

    MultiIter_Done(mi) = d;

    return mi;
}

multi_iter_masked_t* 
multi_iter_masked_new(
    npy_intp shape[], int rank, int mask[], int mask_len) 
{
    multi_iter_masked_t *mi;
    int i;
    char d = 0;

    assert(rank > 0);

    mi = (multi_iter_masked_t *) mkl_malloc(sizeof(multi_iter_masked_t), 64);
    MultiIter_Index(mi) = (npy_intp *) mkl_calloc(rank,  sizeof(npy_intp), 64);
    MultiIter_Shape(mi) = (npy_intp *) mkl_malloc(rank * sizeof(npy_intp), 64);
    memcpy(MultiIter_Shape(mi), shape, rank * sizeof(npy_intp));
    MultiIter_Rank(mi) = rank;

    for(i=0; i < rank; i++) {
	d |= MultiIter_IndexElem(mi, i) >= MultiIter_ShapeElem(mi, i);
	if (d) break;
    }

    MultiIter_Done(mi) = d;

    assert(mask_len > 0);
    MultiIter_MaskLength(mi) = mask_len;
    MultiIter_Mask(mi) = (int *) mkl_malloc(mask_len * sizeof(int), 64);
    memcpy(MultiIter_Mask(mi), mask, mask_len * sizeof(int));

    return mi;
}


void multi_iter_masked_free(multi_iter_masked_t *mi) {
    if (mi) {
	if(MultiIter_Index(mi))
	    mkl_free(MultiIter_Index(mi));

	if (MultiIter_Shape(mi))
	    mkl_free(MultiIter_Shape(mi));

	if (MultiIter_Mask(mi))
	    mkl_free(MultiIter_Mask(mi));

	mkl_free(mi);
    }

    return;
}

void multi_iter_free(multi_iter_t *mi) {
    if (mi) {
	if(MultiIter_Index(mi))
	    mkl_free(MultiIter_Index(mi));

	if (MultiIter_Shape(mi))
	    mkl_free(MultiIter_Shape(mi));

	mkl_free(mi);
    }

    return;
}


/* Modifies iterator in-place, returns 1 when iterator is empty, 0 otherwise */
int multi_iter_next(multi_iter_t *mi) {
    int j, k;

    if(MultiIter_Done(mi)) return 1;
  
    for(k = MultiIter_Rank(mi); k > 0; k--) {
	j = k-1;
	if (++(MultiIter_IndexElem(mi, j)) < MultiIter_ShapeElem(mi, j))
	    return 0;
	else {
	    MultiIter_IndexElem(mi, j) = 0;
	    if (!j) {
		MultiIter_Done(mi) = 1;
	    }
	}
    }

    return 1;
}

/* Modifies iterator in-place, returns 1 when iterator is empty, 0 otherwise */
int multi_iter_masked_next(multi_iter_masked_t *mi) {
    int j, k;

    if(MultiIter_Done(mi)) return 1;

    for(k = MultiIter_MaskLength(mi); k >0; k--) {
	j = MultiIter_MaskElem(mi, k - 1);
	if (++(MultiIter_IndexElem(mi, j)) < MultiIter_ShapeElem(mi, j))
	    return 0;
	else {
	    MultiIter_IndexElem(mi, j) = 0;
	    if (!k) {
		MultiIter_Done(mi) = 1;
	    }
	}
    }

    return 1;
}

