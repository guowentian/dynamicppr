#ifndef __EDGE_BATCH_H__
#define __EDGE_BATCH_H__

#include "Meta.h"

struct EdgeBatch{

    EdgeBatch(IndexType sz){
        size = sz;
        length = 0;
        edge1 = new IndexType[size];
        edge2 = new IndexType[size];
        is_insert = new bool[size];
    }
    ~EdgeBatch(){
        delete[] edge1;
        edge1 = NULL;
        delete[] edge2;
        edge2 = NULL;
        delete[] is_insert;
        is_insert = NULL;
    }

    IndexType *edge1;
    IndexType *edge2;
    bool *is_insert;

    IndexType length;
    IndexType size;
};

#endif
