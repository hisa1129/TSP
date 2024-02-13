import math
import numpy as np
import matplotlib.pyplot as plt

#最適解の確認
def opt_check(dataname):
    root = []
    f = open(dataname)
    lines = f.readlines()
    print (lines)
    datalen = len(lines)
    f.close()
    for i in range(datalen):
        sp = lines[i].split()
        root.append(int(sp[0]) - 1)
    return root

# データの読み込みと初期化
# 引数:
#   dataname: データファイルの名前
# 返り値:
#   data: 都市の座標データ (NumPy配列)
#   datalen: 都市の数
#   root: 初期経路 (リスト)
#   ex_root: 未訪問都市の番号 (リスト)
def init_data(dataname):
    # *** 初期化
    # リスト：順序付きのデータを格納するためのデータ構造です。
    # リストは、要素を追加、削除、変更することができ、さまざまな方法で要素にアクセスできます。
    root = [] # 空のリストを作成　
    ex_root = []
    
    # datanameで指定されたデータファイルを開きます。
    f = open(dataname)
    lines = f.readlines() # .readlines()メソッド: データファイルの内容を1行ずつ読み込み、リストとして返す
    datalen = len(lines) # 都市の数datalenをlinesの行数から取得します。
    print ("lines[0]")
    print (lines[0]) # リスト型linesの0番目　1 565.0 575.0
    print ("lines[51]")
    print (lines[51]) # リスト型linesの51番目　52 1740.0 245.0
    # print (type(lines))
    print (lines)
    
    # 全体　['1 565.0 575.0\n', '2 25.0 185.0\n', '3 345.0 750.0\n', '4 945.0 685.0\n', ...
    f.close()
    
    data = np.zeros((datalen, 2)) # NumPy配列dataを(datalen, 2)の大きさで初期化
    print(data)
    # (datalen, 2)の大きさのNumPy配列dataを作成し、すべての要素を0で初期化しています。
    # datalem = 52の場合、 
    # data = np.zeros((52, 2))
    # [[0. 0.]
    # [0. 0.]
    # [0. 0.]
    # [0. 0.]
    # ...
    # [0. 0.]
    # [0. 0.]]
    #
    # data[0, 0] = 0
    # data[0, 1] = 0
    # ...
    # data[51, 0] = 0
    # data[51, 1] = 0
    
    # *** 都市の座標データが読み込み
    # linesリストに格納された都市座標データを読み込み、data配列に格納する処理
    for i in range (datalen): # iを0からdatalen-1までループ
        ex_root.append(i) # 現在の都市番号iをex_rootリストに追加
        
        # linesリストのi番目の要素を空白文字で分割し、リストspに格納します。
        # split() メソッドは、文字列を空白文字で分割し、リストとして返します。
        sp = lines[i].split() # リスト型linesの0番目の内容:1 565.0 575.0
        
        print ("都市の座標データの分割結果読確認")
        # 配列NO＝読み込み順が都市のNOになっている
        print (sp[0]) #都市NO
        print (sp[1]) #X座標
        print (sp[2]) #Y座標
        data[i,0] = sp[1] # x座標の格納: spリストの2番目の要素をdata配列のi行0列目に格納します。
        data[i,1] = sp[2] # y座標の格納: spリストの3番目の要素をdata配列のi行1列目に格納します。
        # datalem = 52で、データファイル berlin52.txtの内容が以下の場合:
        # 1 565 575
        # 2 25 185
        # ...
        # data[0, 0] = 565
        # data[0, 1] = 575
        # data[1, 0] = 25
        # data[1, 1] = 185
        # ...
        # data[51, 0] = 1740
        # data[51, 1] = 245
        print("data[i]")
        print(data[i])
        
        print("data[i,0]")
        print(data[i,0])
        print("data[i,1]")
        print(data[i,1])

    
    print(data)
    print(datalen)
    print(root)
    print(ex_root)
    return data, datalen, root, ex_root

#グラハムスキャン
def convex(data):

    #y座標最小のものを探す
    min = 0
    for i in range(datalen):
        if (data[min, 1] > data[i, 1]):
            min = i
        elif (data[min, 1] == data[i, 1] and  data[min, 0] < data[i, 0]):
            min = i

    #反時計周りでの角度を調べる
    angle = np.zeros((datalen,2))
    for i in range(datalen):
        if (i == min):
            angle[i] = [0, i]
        else:
            theta = math.atan2((data[i, 1] - data[min, 1]), data[i, 0] - data[min, 0])
            if (theta < 0):
                theta = (2 * math.pi) + theta;
            angle[i] = [theta, i]

    #角度順にソート
    sorted = angle[angle[:,0].argsort(), :]

    stack = []
    stack.extend([sorted[0, 1], sorted[1, 1], sorted[2, 1]])

    for i in range(3, datalen):
        stacktop = len(stack)
        while(True):
            theta1 = math.atan2(data[int(stack[stacktop - 1]), 1] - data[int(stack[stacktop - 2]), 1],
                                data[int(stack[stacktop - 1]), 0] - data[int(stack[stacktop - 2]), 0])
            if (theta1 < 0): theta1 = 2 * math.pi + theta1
            theta2 = math.atan2(data[int(sorted[i, 1]), 1] - data[int(stack[stacktop - 1]), 1],
                                data[int(sorted[i, 1]), 0] - data[int(stack[stacktop - 1]), 0])
            if (theta2 <= 0): theta2 = 2 * math.pi + theta2
            if (theta2 - theta1 < 0):
                del stack[stacktop - 1]
                stacktop -= 1
            else:
                break
        stack.append(sorted[i, 1])

    for i in range (len(stack)):
        stack[i] = int(stack[i])
    return stack

#角度を計算
def angle(x, y):

    dot_xy = np.dot(x, y)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    cos = dot_xy / (norm_x*norm_y)
    rad = np.arccos(cos)
    theta = rad * 180 / np.pi

    return theta

#最近挿入法
def insertion(root, ex_root):

    for i , number in enumerate(root):
        ex_root.remove(number)

    while (True):
        min = 0
        costratio = [0 for i in range(len(root))]
        minNum = [0 for i in range(len(root))]
        for i in range (len(root)):
            for j in range(0, len(ex_root)):
                if j == 0 or min > cal_cost(root[i - 1], root[i], ex_root[j]):
                    min = (cal_cost(root[i - 1], root[i], ex_root[j]))
                    minNum[i] = ex_root[j]
            costratio[i] = cal_costratio(root[i - 1],root[i], minNum[i])

        ratiomin = 9999
        ratiominNum = 0

        for i in range (len(root)):
            if ratiomin > costratio[i]:
                ratiomin = costratio[i]
                ratiominNum = i

        root.insert(ratiominNum, minNum[ratiominNum])
        ex_root.remove(minNum[ratiominNum])

        if not ex_root:
            break

    return root

# Nearest Neighbor法
# 現在の都市から最も近い都市を巡る経路を生成する関数
# 
# Nearest Neighbor法のアルゴリズムは、次のように表されます。
# 1.適当な都市を選び、出発点とする。
# 2.まだ訪れていない都市のうち、現在いる都市から最も近い都市を選び、
#  その都市との経路を巡回路に加え、その都市に移動する。
# 3.2.を、まだ訪れていない都市がなくなるまで繰り返す。
#  なくなったら、最後に訪れた都市と出発点とをつなぐ経路を巡回路に加えて終了する。
# 引数:
#   data: 都市の座標データを格納したNumPy配列
#   datalen: データファイルに含まれる都市の数
#   root: 現在の経路
#   ex_root: 未訪問都市の番号
# 返り値:
#   root: 巡る都市の番号のリスト
def nearest_n(data, datalen, root, ex_root):
    # 初期化:
    # 現在の都市として、まず最初に配列の0番目の都市をrootリストに追加します。
    # 0番目の都市をex_rootリスト（未探索都市リスト）から削除します。
    root.append(0) # 探索した都市、訪問順
    ex_root.remove(0) #まだ訪問していない都市のNOから0番目をののぞう

    # 現在の都市から最も近い都市を探し、rootリストに追加します。
    # 最も近い都市をex_rootリストから削除します
    #
    # 0番目の都市からdatalen-2番目の都市までのすべての都市について、最も近い都市を探します。
    # datalen-1番目の都市は、最後の都市なので、このループで処理する必要はありません。
    #
    # 現在の都市とは、rootリストの一番後ろの要素
    # 現在の都市に最も近い都市が追加されると、rootリストの最後の要素が変化します。
    # よって、現在の都市は、rootリストの最後の要素が変化するたびに変化します。
    for i in range(datalen - 1):
        # range(datalen - 1)は、0からdatalen - 2までの数値を生成
        # 最小距離の探索
        print(i)
        min_len = 0 # 現在の都市までの最小距離を格納します
        min_Num = 0 # 現在の都市から最も近い都市の番号を格納します
        print("現在の都市NO")
        print(root)
        print("計算対象の都市NO")
        print(ex_root)
        # ex_rootリスト内のすべての都市について、現在の都市(rootリストの要素)との距離を計算します
        for j in range(len(ex_root)):
            print(j)
            print("都市NO")
            print("現在の都市NO")
            print (root[i])
            print("計算対象都市全体")
            print(ex_root)
            print("計算対象の都市NO")
            print (ex_root[j])

            print("新たに計算した都市までの距離を求める")
            print("都市の座標データ全体")
            print(data)
            print("現在の都市の座標")
            print(data[root[i]] )
            print("計算対象の都市の座標")
            print(data[ex_root[j]] )
            # np.linalg.norm([data[root[i]] - data[ex_root[j]]])は、
            # 2点間data[root[i]＝(x1,y1),data[ex_root[j]＝(x2,y2)の距離を計算する
            # 現在の都市root[i]とex_rootリスト内の都市jとの距離を計算します。
            # ２つの都市の座標間の距離をnumpyのノルムを使って計算する
            print("２つの都市の座標間の距離をnumpyのノルムを使って計算する")
            new_length = np.linalg.norm([data[root[i]] - data[ex_root[j]] ])
            print(new_length)
            
            print("現時点の最小距離")
            print(min_len)
            # 計算結果がより小さい場合、現在の都市から最も近い都市を入れ替える
            if j == 0 or min_len > new_length:
                print("入れ替え処理")
                min_len = new_length
                min_Num = ex_root[j]
                print(min_len)
                print(min_Num)
            print("最小距離")
            print(min_len)
            print("現在の都市から計算した中で近い都市")
            print(min_Num)
            print("ーーーーーーーーーーーーー")
            
        # min_Numは、現在の都市から最も近い都市の番号です。この都市をrootリストに追加することで、経路を更新します
        print("現在の都市NO")
        print (root[i])
        print("ここから一番近い都市NO")
        print(min_Num)
        root.append(min_Num) # 経路への追加:最も近い都市min_Numをrootリストに追加します
        ex_root.remove(min_Num) # 未訪問都市リストの更新:最も近い都市min_Numをex_rootリストから削除します
    print(root)
    print(ex_root)
    return root

#2-opt法
# 2-opt法
# 2-opt 法は逐次改善法の一種であり、逐次改善法とは、ある巡回路を基として、 
# それより更にコストの小さい巡回路を探す方法です。 
# 2-opt 法のアルゴリズムは、次のように表されます。
# 
# 1.入力した巡回路に対し、適当な二つの辺を選択し、それらを入れ替えた結果のコストを計算する。 
#  そのコストが入れ替える前より小さくなれば、入れ替えを採用し、巡回路を改善する。
# 1.を、巡回路の改善ができなくなるまで繰り返す。
# 
# 逐次改善法は既存の巡回路を改善していくことで最適解に近づけていくものなので、 
# 事前に初期解として何らかの巡回路を与えておかなければなりません。 
# 適当に巡回路を定めても良いのですが、ここでは構築法(与えられた都市間のコストから、巡回路を構築していく方法) である
# Nearest Neighbor法、Convex Hull Insertion法の2つを実装しました。
# 引数:
#   data: 都市の座標データを格納したNumPy配列
#   datalen: データファイルに含まれる都市の数
#   root: 現在の経路
# 返り値:
#   root: 2-opt法で改善された経路
#
def opt_2(data, datalen, root):
    print("現在の経路")
    print(root)
    total = 0
    # 経路が改善されなくなるまでループを続けます。
    while True:
        count = 0
        # 辺の入れ替え
        # iとjは、入れ替える2つの辺のインデックスを表します。
        # ループ１周目の例
        # i = 0
        # j = 2
        # l1=(i, i1): root[0] と root[1] 間の距離
        # l2=(j, j1): root[2] と root[3] 間の距離
        # l3=(i, j): root[0] と root[2] 間の距離
        # l4=(i1, j1): root[1] と root[3] 間の距離
        #
        # datalen=10の場合、
        # for i in range(datalen - 2):
        #    for j in range(i + 2, datalen):
        # 外側のループは range(datalen - 2) で定義されているため、
        # iの値は0から(datalen - 2) -1) まで変化します。これは0から7までの8つの値です。
        # 内側のループは range(i + 2, datalen) で定義されており、
        # iの値に依存しています。
        # 具体的には、i + 2から(datalen-1)までの値を取ります。
        # したがって、内側のループでは、iの値に2を加えた値からdatalenまでの値がjの値として取られます。  
        # iは、0から７まで８回回る
        # i = 0 の場合: j = 2, 3, 4, 5, 6, 7, 8, 9
        # i = 1 の場合: j = 3, 4, 5, 6, 7, 8, 9
        # i = 2 の場合: j = 4, 5, 6, 7, 8, 9
        # i = 3 の場合: j = 5, 6, 7, 8, 9
        # i = 4 の場合: j = 6, 7, 8, 9
        # i = 5 の場合: j = 7, 8, 9
        # i = 6 の場合: j = 8, 9
        # i = 7 の場合: j = 9
        #
        # range(n)は、0からn−１の値を生成する
        # range(0, 5) --> 0 1 2 3 4
        # range(a,n)は、aからn-1の値を生成する
        # range(4,7) --> 4 5 6
        # 
                
                
        for i in range(datalen - 2):
            i1 = i + 1
            for j in range(i + 2, datalen):
                # jは、i+2からスタートする
                if j == datalen - 1:
                    j1 = 0
                else:
                    j1 = j + 1
                print("i")
                print("i1")
                print("j")
                print("j1")
                print(i)
                print(i1)
                print(j)
                print(j1)
                if i != 0 or j1 != 0:
                    # 辺の長さを計算
                   
                    # ２つの都市の座標間の距離をnumpyのノルムを使って計算する
                    # np.linalg.norm([data[root[i]] - data[ex_root[j]]])は、
                    # 2点間data[root[i]＝(x1,y1),data[ex_root[j]＝(x2,y2)の距離を計算する
                    print("２つの都市の座標間の距離をnumpyのノルムを使って計算する")
                    print("i")
                    print("i1")
                    print("j")
                    print("j1")
                    print(i)
                    print(i1)
                    print(j)
                    print(j1)
                    # l1とl2は、元の経路における2つの辺の長さを表します。
                    print(data[root[i]])
                    print(data[root[i1]])
                    print([ data[root[i]] - data[root[i1]] ])
                    #print((data[root[i]] - data[root[i1]] )
                    l1 = np.linalg.norm([data[root[i]] - data[root[i1]]])
                    print(l1)
                    
                    print(data[root[j]])
                    print(data[root[j1]])
                    print([ data[root[j]] - data[root[j1]] ])
                    l2 = np.linalg.norm([data[root[j]] - data[root[j1]]])
                    print(l2)
                    
                    # l3とl4は、辺を入れ替えた後の経路における2つの辺の長さを表します。
                    print(data[root[i]])
                    print(data[root[j]])
                    print([ data[root[i]] - data[root[j]] ])
                    l3 = np.linalg.norm([data[root[i]] - data[root[j]]])
                    print(l3)
                    
                    print(data[root[i1]])
                    print(data[root[j1]])
                    print([ data[root[i1]] - data[root[j1]] ])
                    l4 = np.linalg.norm([data[root[i1]] - data[root[j1]]])
                    print(l4)
                    print("")
                    print(l1+l2)
                    print(l3+l4)
                    # 入れ替えた方が短くなる場合
                    if l1 + l2 > l3 + l4:
                        print("入れ替えた方が短くなる場合")
                        # 辺を入れ替える
                        # new_rootは、i1からjまでの部分リストを逆順にしたリストです。
                        print(root)
                        print(i1)
                        print(root[i1])
                        print(j+1)
                        # print(root[j+1])
                        new_root = root[i1:j+1]
                        print(new_root)
                        # root[i1:j+1]は、new_rootで置き換えられます。
                        print(new_root[::-1])
                        root[i1:j+1] = new_root[::-1]
                        print(root)
                        
                        count += 1
        total += count
        print (root)
        if count == 0: break

    return root

#追加コストを計算
def cal_cost(i,j,k):
    return np.linalg.norm([data[i] - data[k]]) + np.linalg.norm([data[k] - data[j]])\
           - np.linalg.norm([data[i] - data[j]])

#コスト比を計算
def cal_costratio(i,j,k):
    return (np.linalg.norm([data[i] - data[k]]) + np.linalg.norm([data[k] - data[j]])) / np.linalg.norm([data[i] - data[j]])

#総コスト計算
def cal_totalcost(data, root):
    totalcost = 0
    for i in range(len(root)):
        totalcost += np.linalg.norm(([data[root[i]] - data[root[i-1]]]))
    return totalcost

#図にプロット
def autoplot(root):
    plt.scatter(data[:, 0], data[:, 1])
    initnum = 0
    beforenum = 0
    for i, number in enumerate(root):
        plt.scatter(data[int(number), 0], data[int(number), 1], c='red')
        if i == 0:
            beforenum = number
            initnum = number
        else:
            plt.plot([data[int(beforenum), 0], data[int(number), 0]], [data[int(beforenum), 1], data[int(number), 1]], 'r')
            beforenum = number

    plt.plot([data[int(beforenum), 0], data[int(initnum), 0]], [data[int(beforenum), 1], data[int(initnum), 1]], 'r')
    plt.show()
    plt.close()

if __name__ == '__main__':

    dataname = 'berlin52.txt'
    
    type = 3
    # 1: NN法（Nearest Neighbor法）
    # 2: CHI法 Convex Hull Insertion法
    # 3: NN法＋2-opt法、
    # 4: CHI法＋2-opt法
    
    # data: 都市の座標データ。各都市は2次元配列の1行に格納され、1列目はx座標、2列目はy座標を表します。
    # datalen: 都市の数(=データファイルの行数)
    # root: 初期経路。都市の番号を順番に格納した配列です。0からdatalen-1までの数字が順番に格納されます。(これは、初期経路がまだ決まっていないことを意味します。)
    # ex_root: 未訪問都市の番号を格納した配列です。root配列に含まれていない数字が全て格納されます。(これは、全ての都市が未訪問であることを意味します。)
    data, datalen, root, ex_root = init_data(dataname)
    ''' 
    返り値が複数ある場合は、タプルになる（pythonの仕様）
    これと同じ
    ret = init_data(dataname)
    data = ret[0]
    datalen = ret[1]
    root = ret[2]
    ex_root = ret[3]
    '''

    print("読み込み確認")
    print(data)
    print("")
    print(datalen)
    print("")
    print(root)
    print("")
    print(ex_root)


    if type == 1:
        root = nearest_n(data, datalen, root, ex_root)
    elif type == 2:
        root = convex(data)
        root = insertion(root, ex_root)
    elif type == 3:
        root = nearest_n(data, datalen, root, ex_root)
        root = opt_2(data, datalen, root)
    elif type == 4:
        root = convex(data)
        root = insertion(root, ex_root)
        root = opt_2(data, datalen, root)

    print (cal_totalcost(data, root))
    autoplot(root)




