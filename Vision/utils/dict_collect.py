

def find_dic(item, key):
    if isinstance(item, dict):
        for k, v in item.items():
            key[k] = v
            find_dic(v, key)
    else:
        return item


def get_by_key(item, key):
    dic = {}
    find_dic(item, dic)
    value = dic.get(key)
    return value


if __name__ == "__main__":
    d = {'name': 'xiaohong',
         'famliy_member': {'self': 'xiaohong', 'children': {'son': 'xiaoxiaobai', 'g': 'xiaoxiaohong'}}}
    v = get_by_key(d, 'children')
    print(v)
