

      
# -*- coding: utf-8 -*-
"""
素数集合及びCramér型集合 × 貪欲法 汎用実験スクリプト
動作環境: Python 3 (PC、Pythonista 3 (iOS) のどちらでも動作)
    ・計算部分は標準ライブラリのみで動作します。
    ・グラフ表示にだけ matplotlib が必要です (pip install matplotlib)。
      未導入の場合は、グラフ表示だけがスキップされます。
    ・グラフのウィンドウは、閉じると次の処理に進みます。
    ・実行: python Mathlog.py  (判定対象の最大値などを対話式で入力します)
    ・実行時間・メモリの目安 (環境により変わります):
        判定対象の最大値 10^6 → 約20秒・約25MB
        判定対象の最大値 10^7 → 約10分・約160MB
      それ以上はさらに長時間かかるため、まず 10^6 程度でお試しください。
このスクリプトの狙い:
    奇数列(プール)を
        1) 素数列そのまま (3以上の素数)
        2) クラメール型奇数列 (局所素数密度 x density_scale の確率で奇数を間引く。
           seedで「どの奇数が生存候補になるか」の選び方も変えられる)
    のどちらかで作り、貪欲法を適用して、ゴールドバッハ強度(生存要素数・穴の有無など)を比較できるようにする。
    貪欲法は (elements_full, is_member, candidates) という3点セット(プール)さえ
    渡せば動く汎用エンジンになっている。
    密度の校正は固定幅ではなく、Nに応じて可変の適応的区間幅で行う
    (固定幅だと大きなNで区間内素数個数が減りノイズが乗るため)。
    篩処理後は「表現数プロット」(範囲を変えて繰り返し可)が使える。
    表現数プロットには、表現数1〜3の偶数の割合(区間ごと・対数軸)のグラフと、
    コンソールへの個数・割合の表示が付く。
"""
import sys
import time
import math
import random
import bisect
def _print_progress(done, total, t0, label="進捗"):
    if total <= 0:
        return
    pct = done / total * 100.0
    elapsed = time.time() - t0
    if done > 0 and elapsed > 0:
        eta = elapsed / done * (total - done)
        sys.stdout.write("\r{}: {:5.1f}% ({}/{})  経過:{:.1f}秒  残り目安:{:.1f}秒   ".format(
            label, pct, done, total, elapsed, eta))
    else:
        sys.stdout.write("\r{}: {:5.1f}% ({}/{})   ".format(label, pct, done, total))
    sys.stdout.flush()
def _finish_progress_line():
    sys.stdout.write("\n")
    sys.stdout.flush()
def print_number_list(numbers, label="", max_show=200, per_line=10):
    """
    数値リストを見やすく表示する。
    件数がmax_showを超える場合は先頭・末尾のみ表示し、中間は省略する
    (件数自体は常に表示するので、全件必要な場合は返り値のリストを別途参照/保存すること)。
    """
    numbers = list(numbers)
    print("\n=== {} (件数: {}) ===".format(label, len(numbers)))
    if not numbers:
        print("(空)")
        return
    if len(numbers) <= max_show:
        display = [str(x) for x in numbers]
    else:
        half = max_show // 2
        display = ([str(x) for x in numbers[:half]] +
                   ["... ({}件省略) ...".format(len(numbers) - max_show)] +
                   [str(x) for x in numbers[-half:]])
    line = []
    for i, s in enumerate(display, 1):
        line.append(s)
        if i % per_line == 0:
            print(" ".join(line))
            line = []
    if line:
        print(" ".join(line))
# ============================================================
# 1. 基礎: 篩・素数リスト・接頭辞和(局所密度算出用)
# ============================================================
def sieve(limit):
    """エラトステネスの篩。is_p[i]==1ならiは素数"""
    limit = max(limit, 2)
    is_p = bytearray([1]) * (limit + 1)
    is_p[0] = 0
    is_p[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if is_p[i]:
            is_p[i * i::i] = bytearray(len(range(i * i, limit + 1, i)))
    return is_p
def build_prime_prefix(is_p):
    """prefix[i] = is_p[0..i]内の素数個数(累積和)。局所素数密度をO(1)で求めるため。"""
    n = len(is_p)
    prefix = [0] * n
    running = 0
    for i in range(n):
        if is_p[i]:
            running += 1
        prefix[i] = running
    return prefix
def prime_count_in_range(prefix, lo, hi):
    """[lo, hi] (両端含む)内の素数個数。範囲外は自動的にクリップする。"""
    n = len(prefix)
    hi = min(hi, n - 1)
    lo = max(lo, 0)
    if hi < lo:
        return 0
    lo_prev = prefix[lo - 1] if lo > 0 else 0
    return prefix[hi] - lo_prev
# ============================================================
# 2. 候補プールの生成方法いろいろ
#    (どれも最終的に (elements_full, is_member, candidates) を返す)
# ============================================================
def generate_prime_sequence(limit, buffer=1000, min_p=3):
    """limit以下の素数列(min_p以上。既定は3から。2は基礎数として別途プールに加える)"""
    is_p = sieve(limit + buffer)
    return [p for p in range(min_p, limit + 1) if is_p[p]]
def default_adaptive_window(n):
    """
    局所密度を測るための適応的半径(窓の半分の幅)。
    Nが大きいほど窓を広げて推定を安定させる。
    (ここはチューニング対象。window_fn引数で好きな関数に差し替え可能)
    """
    return max(50, int(n ** 0.5))
def generate_density_thinned_odds(limit, density_scale=1.0, seed=0, buffer=1000,
                                   window_fn=None, min_v=3, fixed_survivors=(3, 5, 7),
                                   verbose=False):
    """
    奇数列を、局所的な素数密度(適応的区間幅で算出)× density_scale の確率で
    独立に間引いて生き残らせた「密度可変odd列」を生成する。
    考え方:
        「局所的な素数密度と同じ確率で各奇数を独立に生かすかどうか
        コイン投げする」というCramér流のモデルを使えば、
        density_scale=1.0 で素数相当の密度になり、
        density_scale を変えれば密度そのものを、seed を変えれば
        「同じ密度でもどの奇数が生き残るか」という選択パターンを
        それぞれ独立にコントロールできる。
    fixed_survivors: この奇数だけは乱数によらず必ず生存させる(既定: 3, 5, 7)。
                      それ以外は他の奇数と同様に確率的間引きの
                      対象になる(=穴が開くケースも含めてデータを取る狙い)。
    density_scale: 密度の倍率。1.0で素数密度相当
    seed:          乱数シード。生存奇数の選び方(実現値)を変えるのに使う。
    window_fn(n):  局所密度推定用の窓の半径を返す関数。省略時はdefault_adaptive_window。
    """
    rng = random.Random(seed)
    sieve_limit = limit + buffer
    is_p = sieve(sieve_limit)
    prefix = build_prime_prefix(is_p)
    if window_fn is None:
        window_fn = default_adaptive_window
    fixed_set = set(fixed_survivors)
    seq = []
    n = min_v if min_v % 2 == 1 else min_v + 1
    total = max(1, (limit - n) // 2 + 1)
    done = 0
    t0 = time.time()
    report_every = max(1, total // 200)
    while n <= limit:
        if n in fixed_set:
            seq.append(n)
        else:
            w = window_fn(n)
            lo, hi = max(0, n - w), min(sieve_limit, n + w)
            odd_count_in_window = (hi - lo) // 2 + 1
            prime_count = prime_count_in_range(prefix, lo, hi)
            local_density_per_odd = (prime_count / odd_count_in_window) if odd_count_in_window > 0 else 0.0
            p = min(1.0, local_density_per_odd * density_scale)
            if rng.random() < p:
                seq.append(n)
        n += 2
        done += 1
        if verbose and (done % report_every == 0 or done == total):
            _print_progress(done, total, t0, label="density間引き")
    if verbose:
        _finish_progress_line()
    return seq
def build_elements_list(seq, include_base_2=True):
    if include_base_2:
        return sorted(set([2] + list(seq)))
    return sorted(set(seq))
def build_membership_array(seq, limit, include_base_2=True):
    arr = bytearray(limit + 1)
    for x in seq:
        if x <= limit:
            arr[x] = 1
    if include_base_2 and limit >= 2:
        arr[2] = 1
    return arr
def build_pool_from_seq(seq, sieve_limit, include_base_2=True):
    """
    任意の候補列(seq)から、貪欲法エンジンが必要とする
    (elements_full, is_member, candidates) の3点セットを作る。
    sieve_limitはis_memberの配列サイズ(rを生存要素に加える操作の余裕分を含めた上限)。
    """
    elements_full = build_elements_list(seq, include_base_2=include_base_2)
    is_member = build_membership_array(seq, sieve_limit, include_base_2=include_base_2)
    return elements_full, is_member, list(seq)
def build_prime_pool(limit, buffer=1000):
    """素数列(3以上)からプールを作る"""
    seq = generate_prime_sequence(limit, buffer=buffer)
    return build_pool_from_seq(seq, limit + buffer, include_base_2=True)
def build_density_pool(limit, density_scale=1.0, seed=0, buffer=1000, window_fn=None,
                        fixed_survivors=(3, 5, 7), verbose=False):
    """新規: クラメール型奇数列からプールを作る"""
    seq = generate_density_thinned_odds(limit, density_scale=density_scale, seed=seed,
                                         buffer=buffer, window_fn=window_fn,
                                         fixed_survivors=fixed_survivors, verbose=verbose)
    return build_pool_from_seq(seq, limit + buffer, include_base_2=True)
def build_pool_for_config(limit, config, verbose=False):
    """configのtype("prime" / "density")に応じて、limitまでのプールを作る"""
    buffer = config.get("buffer", 1000)
    ptype = config.get("type")
    if ptype == "prime":
        return build_prime_pool(limit, buffer=buffer)
    elif ptype == "density":
        return build_density_pool(
            limit, density_scale=config.get("density_scale", 1.0),
            seed=config.get("seed", 0), buffer=buffer,
            window_fn=config.get("window_fn"),
            fixed_survivors=config.get("fixed_survivors", (3, 5, 7)), verbose=verbose)
    raise ValueError("未知のtype: {}".format(ptype))
def extend_pool_for_config(config, base_candidates, base_limit, new_limit, buffer,
                            verbose=False):
    """
    篩処理に使ったプール(base_limitまで)を new_limit まで拡張し、
    (elements_full, is_member) を返す。
    density型は、new_limitで生成し直すと窓のクリップ位置が変わるため、
    base_limit付近(Nが大きいとき)で元のプールと生存奇数が食い違う可能性がある。
    そこで base_limit以下は篩処理に使った元の候補(base_candidates)をそのまま使い、
    base_limitより大きい部分だけ新しく生成した列を採用する。
    (乱数は奇数ごとに必ず1回ずつ消費されるので、乱数列自体はずれない)
    """
    if config.get("type") == "prime":
        elements_full, is_member, _ = build_prime_pool(new_limit, buffer=buffer)
        return elements_full, is_member
    seq_ext = generate_density_thinned_odds(
        new_limit, density_scale=config.get("density_scale", 1.0),
        seed=config.get("seed", 0), buffer=buffer,
        window_fn=config.get("window_fn"),
        fixed_survivors=config.get("fixed_survivors", (3, 5, 7)), verbose=verbose)
    seq = [x for x in base_candidates if x <= base_limit] + \
          [x for x in seq_ext if x > base_limit]
    elements_full, is_member, _ = build_pool_from_seq(seq, new_limit + buffer,
                                                       include_base_2=True)
    return elements_full, is_member
# ============================================================
# 3. 貪欲法エンジン(プールの作り方に依存しない汎用版)
# ============================================================
def build_availability(elements, is_member, excluded):
    avail = bytearray(is_member)
    for e in excluded:
        if e < len(avail):
            avail[e] = 0
    available_sorted = [x for x in elements if x < len(avail) and avail[x]]
    return avail, available_sorted
def has_goldbach_rep(m, available_sorted, avail):
    half = m // 2
    for q in available_sorted:
        if q > half:
            break
        r = m - q
        if r < len(avail) and avail[r]:
            return True
    return False
def count_goldbach_reps(m, available_sorted, avail):
    count = 0
    half = m // 2
    for q in available_sorted:
        if q > half:
            break
        r = m - q
        if r < len(avail) and avail[r]:
            count += 1
    return count
def verify_no_holes(excluded, upper_limit, elements, is_member, start_m=4,
                     show_progress=True, progress_label="最終検証"):
    avail, available_sorted = build_availability(elements, is_member, excluded)
    m = start_m if start_m % 2 == 0 else start_m + 1
    if m < 4:
        m = 4
    holes = []
    total = max(1, (upper_limit - m) // 2 + 1)
    done = 0
    t0 = time.time()
    report_every = max(1, total // 200)
    while m <= upper_limit:
        if not has_goldbach_rep(m, available_sorted, avail):
            holes.append(m)
        m += 2
        done += 1
        if show_progress and (done % report_every == 0 or done == total):
            _print_progress(done, total, t0, label=progress_label)
    if show_progress:
        _finish_progress_line()
    return (len(holes) == 0), holes
def check_pool_hole_free(max_v, elements_full, is_member, start_m=4,
                          show_progress=False, label=""):
    """
    篩処理を一切行わず、候補プール全体(is_member)がそのまま
    「全要素利用可能」だった場合に、start_m~max_vの偶数全てに
    ゴールドバッハ的表現が存在するかを確認する(篩処理前の事前穴チェック)。
    貪欲法は候補プールの要素を生存要素に加えるだけで、
    候補プールに元々存在しない組み合わせを新たに作り出すことはできない。
    したがって、この時点で穴が見つかる奇数列は、篩処理を行っても
    その穴を修復できないため、篩処理の対象から外す。
    """
    empty_excluded = set()
    return verify_no_holes(empty_excluded, max_v, elements_full, is_member,
                            start_m=start_m, show_progress=show_progress,
                            progress_label="事前穴チェック" + (" [{}]".format(label) if label else ""))
def lightweight_find_erasure_elements_pool(max_v, elements_full, is_member, candidates,
                                            seed_element=None, verbose=True, label=""):
    """
    貪欲法の汎用エンジン。任意の候補奇数列(プール)に適用できる。
    elements_full, is_member: build_pool_from_seq等で作った判定配列一式
    candidates: 3以上max_v以下の候補要素(順不同で可)
    seed_element: 初期生存要素。省略時は候補の最小値を使う
    処理:
        1. seed_element以外の候補を全て生存要素以外の要素にする
        2. scope初期値(used_qの初期値) = seed_element
        3. 偶数mを6から昇順に検証する
        4. 生存要素qを昇順に走査し、m = q + r を満たすrを探す
             - rが生存要素なら直ちにmの検証を終了(次のmへ)
             - rが生存要素でないなら、最初に見つかったものを生存要素に加える候補として記憶
        5. 生存要素ペアが見つからなければ生存要素に加える候補を生存要素に加え、
           対応するqがscope最大値より大きければscope最大値を更新する
    追記:
        生存要素に加えるrのペアとして使われたq(重複なし)をused_qに記録し、戻り値の末尾で返す。
        scope最大値 : used_qの最大値(初期値=seed_element)。
        scope :「3〜scope最大値の生存要素」の集合
        marginの計算にはscopeではなくused_qの個数を使う。
    """
    candidates = sorted(x for x in set(candidates) if 3 <= x <= max_v)
    if not candidates:
        raise ValueError("候補要素が空です(max_vが小さすぎるか、間引きすぎています)")
    if seed_element is None:
        seed_element = candidates[0]
    elif seed_element not in candidates:
        candidates = sorted(candidates + [seed_element])
    survivors_sorted = [seed_element]
    survivors_set = {seed_element}
    excluded = set(candidates) - {seed_element}
    scope = seed_element
    used_q = set()
    t0 = time.time()
    total = max(1, (max_v - 6) // 2 + 1)
    done = 0
    report_every = max(1, total // 200)
    prog_label = "篩い進捗" + (" [{}]".format(label) if label else "")
    interrupted = False
    interrupted_at = None
    m = 6
    while m <= max_v:
        revival = None
        found_pair = False
        for q in survivors_sorted:
            if q >= m:
                break
            r = m - q
            if r < len(is_member) and is_member[r]:
                if r in survivors_set:
                    found_pair = True
                    break
                elif revival is None and r in excluded:
                    revival = (q, r)
        if not found_pair:
            if revival is None:
                interrupted = True
                interrupted_at = m
                if verbose:
                    _finish_progress_line()
                    print("警告: m={} の修復に使用されたqが見つかりません。処理を中断します。".format(m))
                break
            q, r = revival
            excluded.discard(r)
            survivors_set.add(r)
            bisect.insort(survivors_sorted, r)
            used_q.add(q)
            if q > scope:
                scope = q
        m += 2
        done += 1
        if verbose and (done % report_every == 0 or done == total):
            _print_progress(done, total, t0, label=prog_label)
    if verbose:
        _finish_progress_line()
        status_str = "中断(m={})".format(interrupted_at) if interrupted else "完了"
        print("--- {} [{}] (経過時間:{:.2f}秒, 使用されたqの最大値:{}, 生存要素累計:{}) ---".format(
            status_str, label or "-", time.time() - t0, scope, len(survivors_set)))
    survivor_list = sorted(survivors_set)
    return survivor_list, excluded, scope, candidates, interrupted, interrupted_at, used_q
# ============================================================
# 3.4 表現数プロット
# ============================================================
def restore_tail_for_extended_range(excluded, survivor_list, max_v):
    """
    max_vより後ろの範囲で表現数プロットを行う際に使う。
    生存要素の最大値からmax_vまでの要素は「m<=max_vの検証でたまたま生存要素に加える処理が
    不要だった」だけで、max_v直後のmに対する表現可能性は未検証。
    そのまま生存要素でない扱いにすると不用意な穴が出るため、この範囲は生存側に加える。
    """
    if not survivor_list:
        return set(excluded)
    max_survivor = survivor_list[-1]
    return {e for e in excluded if not (max_survivor < e <= max_v)}
def _plot_label(label):
    """Plot-only label conversion: Japanese pool labels -> English."""
    return str(label).replace("素数", "Primes")
def _import_pyplot():
    """matplotlib.pyplotを読み込む。未導入ならメッセージを出してNoneを返す(グラフだけスキップ)。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlibが見つからないため、グラフ表示をスキップします (pip install matplotlib で導入できます)。")
        return None
    return plt
def _rep_fraction_bins(xs, ys, bin_size):
    """
    偶数の列(xs)とその表現数(ys)を、連続する bin_size 個ずつの区間に分け、
    各区間で「表現数が k (=1,2,3) の偶数の割合」を求める。
    (割合なので区間幅で割る必要はなく、端数の最終区間でも崖状には落ちない。
     ただし偶数が少なすぎる最終区間はノイズが大きいので、bin_size/2 未満なら除外する)
    割合が0の区間は nan にして、対数軸でも線が途切れるだけで済むようにする。
    """
    n = len(ys)
    centers = []
    fractions = {1: [], 2: [], 3: []}
    for i in range(0, n, bin_size):
        seg = ys[i:i + bin_size]
        L = len(seg)
        if L < bin_size / 2.0 and centers:
            break
        centers.append((xs[i] + xs[i + L - 1]) / 2.0)
        for k in fractions:
            c = seg.count(k)
            fractions[k].append(c / L if c > 0 else float("nan"))
    return centers, fractions
def _print_rep_summary(xs, ys, max_v=None):
    """表現数0(穴)・1・2・3の偶数の個数と割合をコンソールに表示する。
    プロット範囲が max_v をまたぐ場合は、篩範囲内(m<=max_v)と範囲外(m>max_v)に分けて表示する。"""
    def block(label, sub):
        n = len(sub)
        if n == 0:
            return
        print("  [{}] 偶数 {} 個".format(label, n))
        total_le3 = 0
        for k in (0, 1, 2, 3):
            c = sub.count(k)
            total_le3 += c
            print("    表現数{}: {:>8} 個 ({:8.4f}%)".format(k, c, c / n * 100.0))
        print("    表現数0〜3の合計: {:>4} 個 ({:8.4f}%)".format(total_le3, total_le3 / n * 100.0))
    print("\n=== 表現数の低い偶数の割合 ===")
    if max_v is None:
        block("全範囲", ys)
    elif xs[-1] <= max_v:
        block("篩範囲内 (m<=max_v={})".format(max_v), ys)
    elif xs[0] > max_v:
        block("篩範囲外 (m>max_v={}) ※max_v以降は未篩の要素も使えるため表現数が増える".format(max_v), ys)
    else:
        cut = sum(1 for x in xs if x <= max_v)
        block("篩範囲内 (m<=max_v={})".format(max_v), ys[:cut])
        block("篩範囲外 (m>max_v={}) ※未篩の要素も使えるため表現数が増える".format(max_v), ys[cut:])
def plot_representations(excluded, elements_full, is_member, start, end,
                          show_progress=True, title_prefix="", max_v=None,
                          bin_size=None, show_density=True):
    """
    指定範囲[start, end]の偶数について、生存要素でない要素を使わない表現数をプロットする。
    show_density=True の場合、散布図の下に「表現数=1,2,3 の偶数の割合」を
    区間ごとに集計した対数軸グラフを同じx軸で表示する(表現数1の枯渇の観察用)。
    bin_size: 1区間あたりの偶数の個数。省略時は全体が約60区間になるよう自動決定(最低200個)。
    max_v:    篩処理の上限。範囲内に含まれるときは縦線を引く
              (それより右は未篩の要素も使えるので、表現数が人工的に増える)。
    """
    plt = _import_pyplot()
    if plt is None:
        return
    import matplotlib.lines as mlines
    from matplotlib.ticker import FuncFormatter
    avail, available_sorted = build_availability(elements_full, is_member, excluded)
    start = start if start % 2 == 0 else start + 1
    if start < 4:
        start = 4
    xs, ys, colors = [], [], []
    total = max(1, (end - start) // 2 + 1)
    done = 0
    t0 = time.time()
    report_every = max(1, total // 200)
    m = start
    while m <= end:
        c = count_goldbach_reps(m, available_sorted, avail)
        xs.append(m)
        ys.append(c)
        if m % 6 == 0:
            colors.append("indianred")
        else:
            colors.append("steelblue")
        m += 2
        done += 1
        if show_progress and (done % report_every == 0 or done == total):
            _print_progress(done, total, t0, label="表現数プロット計算")
    if show_progress:
        _finish_progress_line()
    print("プロットデータ計算完了 ({:.1f}秒)".format(time.time() - t0))
    if not xs:
        print("プロット範囲に偶数がありません。")
        return
    average_reps = sum(ys) / len(ys) if ys else 0
    _print_rep_summary(xs, ys, max_v)
    if show_density:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(10, 9), sharex=True,
            gridspec_kw={"height_ratios": [3, 2]})
    else:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = None
    ax1.scatter(xs, ys, s=4, c=colors)
    ax1.set_ylabel("Number of representations")
    title = "{}Representations after sieving ({} - {})\nMean: {:.2f}".format(
        (title_prefix + " ") if title_prefix else "", start, end, average_reps)
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)
    legend_handles = [
        mlines.Line2D([], [], color="steelblue", marker="o", linestyle="None",
                      markersize=6, label="Even, not a multiple of 6"),
        mlines.Line2D([], [], color="indianred", marker="o", linestyle="None",
                      markersize=6, label="Even, multiple of 6"),
    ]
    boundary_in_range = (max_v is not None and xs[0] < max_v < xs[-1])
    if boundary_in_range:
        ax1.axvline(max_v, color="gray", linestyle="--", linewidth=1)
        legend_handles.append(
            mlines.Line2D([], [], color="gray", linestyle="--",
                          label="Sieve limit max_v (unsieved elements also used to the right)"))
    ax1.legend(handles=legend_handles, loc="best", fontsize=9)
    if ax2 is not None:
        n = len(ys)
        if bin_size is None or bin_size < 1:
            bin_size = max(200, int(math.ceil(n / 60.0)))
        bin_size = min(bin_size, n)
        centers, fractions = _rep_fraction_bins(xs, ys, bin_size)
        k_colors = {1: "tab:red", 2: "tab:orange", 3: "tab:green"}
        plotted = False
        for k in (1, 2, 3):
            vals = fractions[k]
            if all(v != v for v in vals):  # 全てnan(該当する偶数が1つもない)
                continue
            ax2.semilogy(centers, vals, "o-", markersize=3, color=k_colors[k],
                         label="Representations = {}".format(k))
            plotted = True
        if boundary_in_range:
            ax2.axvline(max_v, color="gray", linestyle="--", linewidth=1)
        ax2.set_xlabel("Even number m")
        ax2.set_ylabel("Fraction of evens in the bin")
        ax2.set_title("Fraction of evens with 1-3 representations (bin = {} evens; line breaks where the fraction is 0)".format(bin_size))
        ax2.grid(True, which="both", ls="--", alpha=0.4)
        if plotted:
            ax2.legend(loc="best", fontsize=9)
        else:
            ax2.text(0.5, 0.5, "No evens with 1-3 representations in this range",
                     transform=ax2.transAxes, ha="center", va="center")
    else:
        ax1.set_xlabel("Even number m")
    _fmt = FuncFormatter(lambda v, _pos: "{:,.0f}".format(v))
    for _ax in (ax1, ax2):
        if _ax is not None:
            _ax.xaxis.set_major_formatter(_fmt)
    plt.tight_layout()
    print("グラフを表示します。ウィンドウを閉じると次に進みます。")
    plt.show()
def interactive_plot_representations(result):
    """
    run_pool_experimentの結果(result)を使って、範囲を変えながら何度でも
    表現数プロットを行う対話ループ。
    一度作ったプールは使い回し、範囲がプール上限(pool_limit)を超えたときだけ拡張する。
    """
    if not result.get("pre_check_ok") or result.get("excluded") is None:
        print("事前穴チェックNGのため篩処理が行われていません。表現数プロットはスキップします。")
        return
    if _import_pyplot() is None:
        return
    config = result["config"]
    max_v = result["max_v"]
    buffer = result["buffer"]
    excluded = result["excluded"]
    survivor_list = result["survivor_list"]
    elements_full = result["elements_full"]
    is_member = result["is_member"]
    pool_limit = result["pool_limit"]
    label = result.get("label", "")
    first_time = True
    while True:
        if not first_time:
            again = input("\n続けて別の範囲もプロットしますか? (y/n): ")
            if again.strip().lower() != "y":
                break
        first_time = False
        p_start = int(input("プロット範囲 開始(偶数): "))
        p_end = int(input("プロット範囲 終了(偶数): "))
        if p_end > pool_limit:
            print("プール範囲を拡張中(上限:{})...".format(p_end))
            elements_full, is_member = extend_pool_for_config(
                config, result["base_candidates"], max_v, p_end, buffer)
            pool_limit = p_end
        plot_excluded = excluded
        if p_end > max_v:
            plot_excluded = restore_tail_for_extended_range(excluded, survivor_list, max_v)
        bs_in = input("密度グラフの1区間あたりの偶数の個数 (空欄で自動): ").strip()
        bin_size = int(bs_in) if bs_in else None
        plot_representations(plot_excluded, elements_full, is_member, p_start, p_end,
                              title_prefix="[{}]".format(_plot_label(label)),
                              max_v=max_v, bin_size=bin_size)
# ============================================================
# 3.5 生存奇数密度の両対数グラフ
# ============================================================
def compute_loglog_density_bins(survivor_list, max_v, min_v=3, bins_per_decade=10):
    """
    生存奇数リストを対数スケール上で等幅ビニングし、各ビンでの密度
    (そのビン内の生存奇数個数 / ビンの実幅)を計算する。
    最終ビンがmax_vで割り切れず端数幅になる場合、
    そこだけ密度が崖状に落ちて見えるため、その端数ビンは解析対象から除外する。
    """
    survivors_sorted = sorted(x for x in survivor_list if x >= min_v)
    if len(survivors_sorted) < 2 or max_v <= min_v:
        return [], []
    log_lo = math.log10(survivors_sorted[0])
    log_hi = math.log10(max_v)
    bin_width = 1.0 / bins_per_decade
    centers, densities = [], []
    log_start = log_lo
    while log_start < log_hi:
        log_end = log_start + bin_width
        if log_end > log_hi + 1e-9:
            break  # 端数幅になる最終区間は除外
        e0, e1 = 10 ** log_start, 10 ** log_end
        idx0 = bisect.bisect_left(survivors_sorted, e0)
        idx1 = bisect.bisect_left(survivors_sorted, e1)
        count = idx1 - idx0
        width = e1 - e0
        if width > 0 and count > 0:
            centers.append(math.sqrt(e0 * e1))  # 対数中点
            densities.append(count / width)
        log_start = log_end
    return centers, densities
def plot_survivor_density_loglog(series, bins_per_decade=10, title=None, save_path=None):
    """
    series: [{"label": ..., "survivor_list": [...], "max_v": ...}, ...] のリスト。
    各configの生存奇数密度を同一グラフ上に両対数で重ね描きする。
    """
    plt = _import_pyplot()
    if plt is None:
        return
    plt.figure(figsize=(8, 6))
    plotted = False
    for s in series:
        centers, densities = compute_loglog_density_bins(
            s["survivor_list"], s["max_v"], bins_per_decade=bins_per_decade)
        if not centers:
            continue
        plt.loglog(centers, densities, 'o-', markersize=3, label=_plot_label(s.get("label", "")))
        plotted = True
    if not plotted:
        print("グラフ化できるデータがありません(生存要素が少なすぎるか、範囲が狭すぎます)。")
        plt.close()
        return
    plt.xlabel("Even number m")
    plt.ylabel("Local density (survivors / bin width)")
    plt.title(title or "Survivor density (log-log, bins_per_decade={})".format(bins_per_decade))
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
        print("グラフを保存しました: {}".format(save_path))
    print("グラフを表示します。ウィンドウを閉じると次に進みます。")
    plt.show()
# ============================================================
# 4. 比較実験ハーネス
# ============================================================
def run_pool_experiment(max_v, config, verbose=True,
                         show_lists=True, max_list_display=200):
    """
    1つのプール設定(config)についてプール生成 -> 事前穴チェック -> 貪欲法篩を行い、
    結果の統計dictを返す。
    config例:
        {"label": "素数", "type": "prime"}
        {"label": "density c=1.00 seed=1", "type": "density", "density_scale": 1.0, "seed": 1}
    show_lists: Trueの場合、事前穴チェックに使用する奇数リストと、篩処理後の
                生存奇数リスト(中断時はその時点までのリスト)を表示する。
    max_list_display: リスト表示時に先頭・末尾何件まで表示するか(件数自体は常に表示)。
    """
    label = config.get("label", "")
    buffer = config.get("buffer", 1000)
    elements_full, is_member, candidates = build_pool_for_config(
        max_v, config, verbose=verbose)
    pool_limit = max_v  # is_member/elements_fullが実際に要素を含んでいる上限(seqの上限)
    candidates_in_range = [p for p in candidates if 3 <= p <= max_v]
    if show_lists:
        print_number_list(
            sorted(candidates_in_range),
            label="事前穴チェックに使用するプール要素リスト [{}]".format(label),
            max_show=max_list_display)
    # --- 篩処理前の事前穴チェック ---
    # 候補プール全体をそのまま使ってもmax_vまでの偶数に穴が開くなら、
    # 篩処理を行ってもその穴は直せないため、篩処理自体を行わない。
    pre_ok, pre_holes = check_pool_hole_free(
        max_v, elements_full, is_member, show_progress=verbose, label=label)
    if show_lists:
        print_number_list(
            pre_holes,
            label="事前穴チェックで見つかった穴(偶数)リスト [{}]".format(label),
            max_show=max_list_display)
    if not pre_ok:
        if verbose:
            print("--- [{}] 事前穴チェックNG: {}個の穴 (篩処理は行いません) ---".format(
                label, len(pre_holes)))
        return {
            "label": label,
            "config": config,
            "candidate_count": len(candidates_in_range),
            "pre_check_ok": False,
            "pre_check_holes": pre_holes,
            "survivor_count": None,
            "survivor_ratio_pct": None,
            "scope": None,
            "used_q_count": None,
            "margin_pct": None,
            "elements_full": elements_full,
            "is_member": is_member,
            "survivor_list": None,
            "excluded": None,
            "interrupted": None,
            "interrupted_at": None,
            "max_v": max_v,
            "buffer": buffer,
            "pool_limit": pool_limit,
            "base_candidates": candidates_in_range,
        }
    (survivor_list, excluded, scope, candidates_used,
     interrupted, interrupted_at, used_q) = lightweight_find_erasure_elements_pool(
        max_v, elements_full, is_member, candidates_in_range, verbose=verbose, label=label)
    if show_lists:
        status = "中断: m={}の修復に使用されたqが見つからず停止".format(interrupted_at) if interrupted else "完了"
        print_number_list(
            survivor_list,
            label="篩処理後の生存要素リスト [{}] ({})".format(label, status),
            max_show=max_list_display)
        print_number_list(
            sorted(used_q),
            label="使用されたqのリスト [{}]".format(label),
            max_show=max_list_display)
    # margin: 生存要素に加えるrのペアとして実際に使われたqの個数(重複なし)をBとして、(生存数-B)/生存数 (%)
    #         qとして使われなかった生存要素の割合 = 篩の安全マージンの目安
    used_q_count = len(used_q)
    margin_pct = ((len(survivor_list) - used_q_count) / len(survivor_list) * 100.0
                  if survivor_list else 0.0)
    result = {
        "label": label,
        "config": config,
        "candidate_count": len(candidates_used),
        "pre_check_ok": True,
        "pre_check_holes": [],
        "survivor_count": len(survivor_list),
        "survivor_ratio_pct": (len(survivor_list) / len(candidates_used) * 100.0
                                if candidates_used else 0.0),
        "scope": scope,
        "used_q_count": used_q_count,
        "margin_pct": margin_pct,
        "elements_full": elements_full,
        "is_member": is_member,
        "survivor_list": survivor_list,
        "excluded": excluded,
        "interrupted": interrupted,
        "interrupted_at": interrupted_at,
        "max_v": max_v,
        "buffer": buffer,
        "pool_limit": pool_limit,
        "base_candidates": candidates_in_range,
    }
    return result
def run_comparison(max_v, configs, verbose=True,
                    show_lists=True, max_list_display=200,
                    show_density_plot=True, bins_per_decade=10):
    """
    複数のプール設定(configsのリスト)についてrun_pool_experimentを実行し、
    比較表を表示する。戻り値は各resultのリスト。
    show_density_plot: Trueの場合、各config(事前穴チェックOKかつ生存リストがある
                        ものに限る)の生存要素密度を両対数グラフに重ね描きする。
    """
    results = []
    for cfg in configs:
        if verbose:
            print("\n### プール設定: {} ###".format(cfg.get("label", cfg)))
        res = run_pool_experiment(max_v, cfg, verbose=verbose,
                                   show_lists=show_lists, max_list_display=max_list_display)
        results.append(res)
    print("\n=== 結果 (max_v={}) ===".format(max_v))
    for res in results:
        print("\n----------------------------------------")
        print("設定       : {}".format(res["label"]))
        if not res.get("pre_check_ok", True):
            print("プール候補数     : {}".format(res["candidate_count"]))
            print("事前穴チェック: NG ({}個の穴のため篩処理は未実行)".format(
                len(res["pre_check_holes"])))
            continue
        print("プール要素数     : {}".format(res["candidate_count"]))
        print("生存要素数     : {}".format(res["survivor_count"]))
        print("生存率     : {:.3f}%".format(res["survivor_ratio_pct"]))
        print("使用されたqの最大値     : {}".format(res["scope"]))
        print("使用されたqの個数    : {}".format(res["used_q_count"]))
        print("margin     : {:.3f}%".format(res["margin_pct"]))
    print("----------------------------------------")
    if show_density_plot:
        density_series = [
            {"label": r["label"], "survivor_list": r["survivor_list"], "max_v": max_v}
            for r in results
            if r.get("pre_check_ok") and r.get("survivor_list")
        ]
        if density_series:
            plot_survivor_density_loglog(density_series, bins_per_decade=bins_per_decade)
    return results
# ============================================================
# 5. メイン(対話式)
# ============================================================
def main():
    print("=== 素数集合 / Cramér型集合 × 貪欲法篩い 実験 ===")
    max_v = int(input("判定対象の最大値: "))
    print("\nプール種別: 1) 素数集合  2) Cramér型集合(density_scale=1.0 ≒ 素数相当)")
    while True:
        p_choice = input("選択 (1/2): ").strip()
        if p_choice in ("1", "2"):
            break
        print("1か2を入力してください。")
    if p_choice == "1":
        cfg = {"label": "素数", "type": "prime"}
    else:
        c = float(input("density_scale (例: 1.0): "))
        s = int(input("seed (例: 1): "))
        cfg = {"label": "density c={} seed={}".format(c, s),
               "type": "density", "density_scale": c, "seed": s}
    results = run_comparison(max_v, [cfg])
    res = results[0]
    plot_ans = input("\n表現数をプロットしますか? (y/n): ").strip().lower()
    if plot_ans == "y":
        interactive_plot_representations(res)
if __name__ == "__main__":
    main()

    

