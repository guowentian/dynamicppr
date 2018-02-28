_data_sets=['com-youtube.ungraph.bin', 'com-orkut.ungraph.bin', 'soc-pokec-relationships.bin', 'twitter-2010_rev.bin']
_source_features = ['top10']
_log_dir='log'
_source_dir='exp_vids'
_source_vertex_start_index = 3
_source_vertex_end_index = 4
#_source_vertex_start_index = 0
#_source_vertex_end_index = 10


def is_float_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def extract_ppr_keyword(filename, keyword):
    throughput = ""
    with open(filename) as infile:
        for line in infile:
            pos = line.find(keyword)
            if pos >= 0:
                s = line.find(' ', pos)
                t = line.find(' ', s+1)
                if (is_float_number(line[s+1 : t])):
                    throughput = line[s+1 : t]
    if is_float_number(throughput):
        return float(throughput)
    else:
        print "not a float !"
        return 0

def get_source_vertex_ids(filename):
    ids = []
    with open(filename) as infile:
        for line in infile:
            num = str(int(line))
            if (len(num) > 0):
                ids.append(num)
    return ids

def extract_main_loop(extract_function):
    for data_set in _data_sets:
        print "data_set=%s" % data_set
        for source_feature in _source_features:
            source_vertex_file = _source_dir + '/' + data_set + '_' + source_feature + '.txt'
            source_vertex_ids = get_source_vertex_ids(source_vertex_file)
            source_vertex_ids = source_vertex_ids[_source_vertex_start_index:_source_vertex_end_index]
            extract_function(data_set, source_vertex_ids)

def extract_batch_size(data_set, source_vertex_ids):
    print "=== extract_batch_size ==="
    _batch_sizes = ['1', '10', '100', '1000', '10000', '100000', '1000000']
    for batch_size in _batch_sizes:
        throughput_sum = 0
        for source_vertex_id in source_vertex_ids:
            filename = _log_dir + '/' + 'batch_size_' + data_set + '_' + batch_size + '_' + source_vertex_id + '.txt'
            throughput = extract_ppr_keyword(filename, 'ppr_throughput')
            throughput_sum += throughput
        avg_throughput = throughput_sum / len(source_vertex_ids)
        print avg_throughput

def extract_variant(data_set, source_vertex_ids):
    print "=== extract_variant ==="
    _variants = ['0','1','2','3']
    for variant in _variants:
        latency_sum = 0
        for source_vertex_id in source_vertex_ids:
            filename = _log_dir + '/' + 'op_gpu_' + variant + '_' + data_set + '_' + source_vertex_id + '.txt'
            latency = extract_ppr_keyword(filename, 'ppr_latency')
            latency_sum += latency
        avg_latency = latency_sum / len(source_vertex_ids)
        print avg_latency

def extract_epsilon(data_set, source_vertex_ids):
    print "=== extract_epsilon ==="
    _errors = ['1e5', '1e6', '1e7', '1e8', '1e9', '1e10']
    for error in _errors:
        latency_sum = 0
        for source_vertex_id in source_vertex_ids:
            filename = _log_dir + '/' + 'error_gpu_' + error + '_' + data_set + '_' + source_vertex_id + '.txt'
            latency = extract_ppr_keyword(filename, 'ppr_latency')
            latency_sum += latency
        avg_latency = latency_sum / len(source_vertex_ids)
        print avg_latency

def extract_source_features():
    print "=== extract_source_features ==="
    _source_features = ['top10', 'top1000', 'top1000000']
    for data_set in _data_sets:
        print "data_set=%s" % data_set
        for source_feature in _source_features:
            source_vertex_file = _source_dir + '/' + data_set + '_' + source_feature + '.txt'
            source_vertex_ids = get_source_vertex_ids(source_vertex_file)
            source_vertex_ids = source_vertex_ids[_source_vertex_start_index:_source_vertex_end_index]
            latency_sum = 0
            for source_vertex_id in source_vertex_ids:
                filename = _log_dir + '/' + 'source_feature_' + source_feature + '_' + data_set + '_' + source_vertex_id + '.txt'
                latency = extract_ppr_keyword(filename, 'ppr_latency')
                latency_sum += latency
            avg_latency = latency_sum / len(source_vertex_ids)
            print avg_latency
    _source_features = ['top10']

def extract_batch_ratios(data_set, source_vertex_ids):
    print "=== extract_batch_ratios ==="
    _batch_ratios = ['0.01', '0.001', '0.0001']
    for batch_ratio in _batch_ratios:
        latency_sum = 0
        for source_vertex_id in source_vertex_ids:
            filename = _log_dir + '/' + 'batch_ratio_' + batch_ratio + '_' + data_set + '_' + source_vertex_id + '.txt'
            latency = extract_ppr_keyword(filename, 'ppr_latency')
            latency_sum += latency
        avg_latency = latency_sum / len(source_vertex_ids)
        print avg_latency


if __name__ == "__main__":
    extract_main_loop(extract_batch_size)
    extract_main_loop(extract_variant)
    extract_main_loop(extract_epsilon)
    extract_source_features()
    extract_main_loop(extract_batch_ratios)

