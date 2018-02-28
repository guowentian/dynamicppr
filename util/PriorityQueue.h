#ifndef __PRIORITY_QUEUE_H__
#define __PRIORITY_QUEUE_H__

#include <cassert>
#include <iostream>

// max heap
template<typename TPriority>
struct PNode{
	PNode(size_t k, TPriority p){
		key = k;
		pri = p;
	}
	PNode(){}
	size_t key;
	TPriority pri;
};
template<typename TPriority>
class PriorityQueue{
public:
	PriorityQueue(size_t n){
		// <key, TPriority> pair, key is of type size_t, in range of n
        size = n;
		len = 0;
		priority = new TPriority[size + 1];
		index_to_key = new size_t[size + 1];
		key_to_index = new size_t[size + 1];
		for (size_t i = 0; i <= size; ++i){
			// key may be 0-based
            key_to_index[i] = size + 1;
		}
	}

	~PriorityQueue(){
		delete[] priority;
		priority = NULL;
		delete[] index_to_key;
		index_to_key = NULL;
		delete[] key_to_index;
		key_to_index = NULL;
	}

	size_t GetQueueSize(){
		return len;
	}
	void Insert(size_t k, TPriority pri){
		//assert(key_to_index[k] > size);
		key_to_index[k] = ++len;
		index_to_key[len] = k;
		priority[len] = pri;
		MoveUp(len);
        //assert(len <= size);
	}
	void ChangePriority(size_t key, TPriority pri){
		size_t idx = key_to_index[key];
		if (idx > size){
			// if not exist, insert new one
			Insert(key, pri);
			return;
		}
        TPriority old_pri = priority[idx];
        priority[idx] = pri;
        if (CompareElement(pri, old_pri)){
            // new priority is higher
            MoveUp(idx);
        }
        else{
            MoveDown(idx);
        }
	}
	PNode<TPriority> Pop(){
		//assert(len > 0);
		PNode<TPriority> ret(index_to_key[1], priority[1]);
        SwapElement(1, len);
		len--;
		MoveDown(1);
		key_to_index[ret.key] = size + 1;
        return ret;
	}
	void Remove(size_t key){
		size_t idx = key_to_index[key];
        //assert(idx <= size);
		TPriority orig_pri = priority[idx];
        SwapElement(idx, len);
		//assert(len > 0);
		len--;
		if (CompareElement(priority[idx], orig_pri)){
            MoveUp(idx); 
        }
        else{
            MoveDown(idx);
        }
		key_to_index[key] = size + 1;
	}
	PNode<TPriority> Peek(){
		//assert(len > 0);
		PNode<TPriority> ret(index_to_key[1], priority[1]);
		return ret;
	}
    bool Exists(size_t key){
        return key_to_index[key] <= size;
    }
	void Check(){
		for (size_t i = 1; i <= len; ++i){
			size_t lc = LeftChild(i);
			if (lc < len){
				assert(CompareElement(priority[i], priority[lc]));
			}
			size_t rc = RightChild(i);
			if (rc < len){
				assert(CompareElement(priority[i], priority[rc]));
			}
		}
	}

private:
	void MoveUp(size_t idx){
		while (idx > 1){
			size_t pidx = Parent(idx);
			if (CompareElement(priority[idx], priority[pidx])){
				// idx has higher priority
				SwapElement(idx, pidx);
				idx = pidx;
			}
			else{
				break;
			}
		}
	}

	void MoveDown(size_t idx){
		size_t lc = LeftChild(idx);
		size_t rc = RightChild(idx);
		while (lc <= len){
			size_t max_child = lc;
			if (rc <= len && CompareElement(priority[rc], priority[lc])){
				max_child = rc;
			}
			if (CompareElement(priority[max_child], priority[idx])){
				// if max_child has higher priorty, swap, after swap, "max_child" has higher priority than both of the children
				SwapElement(max_child, idx);
				idx = max_child;
			}
			else{
				break;
			}
            lc = LeftChild(idx);
            rc = RightChild(idx);
		}
	}

	inline bool CompareElement(TPriority p1, TPriority p2){
		return p1 > p2;
	}
	void SwapElement(size_t i, size_t j){
		// i, j are index
		size_t ki = index_to_key[i];
		size_t kj = index_to_key[j];
		std::swap(key_to_index[ki], key_to_index[kj]);
		std::swap(index_to_key[i], index_to_key[j]);
		std::swap(priority[i], priority[j]);
	}

	inline size_t LeftChild(size_t i) { return 2 * i; }
	inline size_t RightChild(size_t i) { return 2 * i + 1; }
	inline size_t Parent(size_t i) { return i / 2; }


	TPriority *priority;
	size_t *index_to_key;
	size_t *key_to_index;
	size_t len;
	size_t size;
};

#endif
