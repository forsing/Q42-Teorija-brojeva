#!/usr/bin/env python3

"""
Q42 Teorija brojeva — Dirichlet karakter / Legendreov simbol / Möbius funkcija
(na granici sa period-finding paradigmom, ali STRUKTURALNO različito) — čisto
kvantno.

Paradigma:
  Analitička teorija brojeva proučava multiplikativne strukture nad prstenom
  ℤ kroz karaktere, L-funkcije, aritmetičke funkcije (μ, φ, σ_k, ...).
  Ključne konstrukcije koje Q42 koristi:

    • Legendreov simbol / kvadratni ostatci mod prostog p:
          χ_p(j) = (j / p) = {+1 ako je j QR mod p,
                               −1 ako je j non-QR,
                                0 ako p | j}
      Zadovoljava Eulerov kriterijum:  χ_p(j) ≡ j^{(p−1)/2} (mod p).
      Za prost p = 37, QR podskup ima (p−1)/2 = 18 elemenata:
          QR_{37} = {1, 3, 4, 7, 9, 10, 11, 12, 16, 21, 25, 26,
                     27, 28, 30, 33, 34, 36}

    • Möbius funkcija μ: multiplikativna, sa
          μ(1) = +1
          μ(n) = (−1)^k  ako je n = p_1 · p_2 · ... · p_k  (squarefree, k razl.)
          μ(n) = 0      ako n ima kvadratni faktor
      Kvadrat μ²(n) = 1 ⇔ n je squarefree.

    • Dirichlet karakter stanja (diskretna Fourier-analitička forma):
          |ψ_χ⟩ = (1 / √Z) · Σ_j  χ_p(j) · g_σ(j − j_target) · |j⟩
      gde je g_σ Gaussian lokalnost oko j_target u poziciji. Stanje ima
      "signed" strukturu (+1 za QR, −1 za non-QR, 0 za multiple p-a ili
      nesquarefree vrednosti kada se pomnoži μ²).

  Analogija sa Gauss sumama:
      G(χ_p, a) = Σ_j χ_p(j) · e^{2πi a j / p}
      G(χ_p, 1) = √p   za χ kvadratni realni karakter (Eisenstein teorema).
  Primena kvantne šetnje (SU(64) Chevalley) na |ψ_χ⟩ generiše interferenciju
  sign-paterna karaktera, što je kvantna analogija ostvarenja Gauss sum
  strukture KROZ dinamiku slobodne čestice na 1D lancu.

Mapiranje na loto:
  Za svaku poziciju i ∈ {1..7}:
    1) j_target (strukturalni cilj, nije frekvencija):
           target_i(prev) = prev + (N_MAX − prev) / (N_NUMBERS − i + 2)
           j_target = round(target_i) − i   ∈ [0, 32]
    2) Mapiranje j ↔ realni lotto broj:  num(j) = i + j.
       Karakter/Möbius funkcije se primenjuju na num(j), ne na j — realni
       brojevi iz opsega [1, 39] nose aritmetičku strukturu.
    3) Inicijalni karakter-Gaussian amplitude:
           A(j) = χ_37(num(j)) · μ²(num(j)) · exp(−(j − j_target)² / (2·σ²))
       Za num(j) ∈ {37} imamo χ_37 = 0 → amplitude nula.
       Za num(j) nesquarefree (4, 8, 9, 12, 16, 18, 20, 24, 25, 27, 28, 32, 36)
       imamo μ² = 0 → amplitude nula.
       Preostali brojevi: +A_Gaussian (QR squarefree) ili −A_Gaussian (non-QR
       squarefree).
    4) Inicijalno stanje:  |ψ_0⟩ = (1/√Z) Σ_j A(j) |j⟩
    5) SU(64) Chevalley slobodna kvantna šetnja:
           H_kin = −J · (T̂_+ + T̂_−)
           U_kin = exp(−i · H_kin · t*)        (Lie grupa SU(64))
           |ψ_QW⟩ = U_kin · |ψ_0⟩
       Hopping mixa signed amplitude preko neighbora na 1D lancu, dajući
       interferenciju između (+1) QR i (−1) non-QR sajtova. t* = 1.0,
       J = 1.0 daje light-cone ≈ 2 — blaga disperzija koja čuva lokalnost
       oko j_target ali meša character-sign pattern.
    6) Born sempling:
           P(j) = |⟨j|ψ_QW⟩|²
       Maskovanje: num > prev_pick, num ∈ [i, i+32]; renormalize; rng.choice.

Dijagnostika po poziciji:
  • ⟨j⟩, σ_j                      — mean/std wavepacket-a u j-prostoru
  • qr_mass = Σ_{num(j) ∈ QR}     — težina na kvadratnim ostatcima
  • sqf_mass = Σ_{num(j) squarefree} — težina na squarefree brojevima
  • entropy H(P) = −Σ P log P     — Shannon entropija raspodele

(okruženje): Python 3.11.13, qiskit 1.4.4, macOS M1, seed = 39.
CSV = /data/loto7hh_4602_k32.csv
CSV u celini (S̄ kao info).
DeprecationWarning / FutureWarning se guse.
NQ = 6 qubit-a po poziciji (DIM = 64), reciklirani registar.
"""


from __future__ import annotations

import csv
import math
import random
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# =========================
# Seed
# =========================
SEED = 39
np.random.seed(SEED)
random.seed(SEED)
try:
    from qiskit_machine_learning.utils import algorithm_globals

    algorithm_globals.random_seed = SEED
except ImportError:
    pass


# =========================
# Konfiguracija
# =========================
CSV_PATH = Path("/data/loto7hh_4602_k32.csv")
N_NUMBERS = 7
N_MAX = 39

NQ = 6                              
DIM = 1 << NQ                       # 64
POS_RANGE = 33                      # Num_i ∈ [i, i + 32]

P_PRIME = 37                        # prost broj za Legendre karakter (p < N_MAX)
SIGMA_INIT = 2.5                    # širina Gaussian lokalnosti u init amplitude
J_HOP = 1.0                         # intenzitet Chevalley NN hopping-a
T_STAR = 1.0                        # vreme kvantne šetnje (light-cone ≈ 2·J·t*)


# =========================
# CSV
# =========================
def load_rows(path: Path) -> np.ndarray:
    rows: List[List[int]] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r, None)
        if not header or "Num1" not in header[0]:
            f.seek(0)
            r = csv.reader(f)
            next(r, None)
        for row in r:
            if not row or row[0].strip() == "Num1":
                continue
            rows.append([int(row[i]) for i in range(N_NUMBERS)])
    return np.array(rows, dtype=int)


def sort_rows_asc(H: np.ndarray) -> np.ndarray:
    return np.sort(H, axis=1)


# =========================
# Structural target (bez frekvencije)
# =========================
def target_num_structural(position_1based: int, prev_pick: int) -> float:
    denom = float(N_NUMBERS - position_1based + 2)
    return float(prev_pick) + float(N_MAX - prev_pick) / denom


def compute_j_target(position_1based: int, prev_pick: int) -> Tuple[int, float]:
    target = target_num_structural(position_1based, prev_pick)
    j = int(round(target)) - position_1based
    j = max(0, min(POS_RANGE - 1, j))
    return j, target


# =========================
# Legendreov simbol χ_p(n) = (n / p)
#   Eulerov kriterijum:  n^((p-1)/2) mod p  ∈ {1, p-1}  (≡ +1, −1)
# =========================
def legendre_symbol(n: int, p: int) -> int:
    n_mod = n % p
    if n_mod == 0:
        return 0
    r = pow(n_mod, (p - 1) // 2, p)
    if r == 1:
        return 1
    if r == p - 1:
        return -1
    return 0


# =========================
# Möbius funkcija μ(n)
#   Preko faktorizacije: μ(n) = (−1)^k ako n = p_1 · ... · p_k squarefree,
#                         μ(n) = 0 inače; μ(1) = +1.
# =========================
def mobius(n: int) -> int:
    if n <= 0:
        return 0
    if n == 1:
        return 1
    # Faktorizacija: probaj proste faktore do sqrt(n)
    m = n
    k = 0
    d = 2
    while d * d <= m:
        if m % d == 0:
            # Proveri da li je d² | n (ne-squarefree)
            m //= d
            if m % d == 0:
                return 0
            k += 1
        else:
            d += 1
    if m > 1:
        k += 1
    return -1 if (k % 2 == 1) else 1


# =========================
# QR skup mod p (za dijagnostiku)
# =========================
def quadratic_residues_mod_p(p: int) -> set:
    return {(x * x) % p for x in range(1, p) if (x * x) % p != 0}


QR_SET = quadratic_residues_mod_p(P_PRIME)


# =========================
# Chevalley shift operatori T̂_± (Q41-stilski)
# =========================
def shift_plus(n: int) -> np.ndarray:
    T = np.zeros((n, n), dtype=np.complex128)
    for j in range(n - 1):
        T[j + 1, j] = 1.0
    return T


T_PLUS = shift_plus(DIM)
T_MINUS = T_PLUS.conj().T
H_KIN = -1.0 * (T_PLUS + T_MINUS)


def evolve_unitary(H: np.ndarray, t: float) -> np.ndarray:
    Hh = (H + H.conj().T) / 2.0
    evals, evecs = np.linalg.eigh(Hh)
    D = np.diag(np.exp(-1j * t * evals))
    return evecs @ D @ evecs.conj().T


U_KIN = evolve_unitary(J_HOP * H_KIN, T_STAR)


# =========================
# Inicijalno Dirichlet character stanje
#   A(j) = χ_p(num(j)) · μ²(num(j)) · exp(-(j - j_target)²/(2σ²))
#   num(j) = position_1based + j
# =========================
def build_char_state(
    j_target: int,
    position_1based: int,
    prev_pick: int,
    sigma: float,
) -> np.ndarray:
    psi = np.zeros(DIM, dtype=np.complex128)
    for j in range(DIM):
        num = position_1based + j
        if num < 1 or num > N_MAX:
            continue
        if num <= prev_pick:
            continue
        chi = float(legendre_symbol(num, P_PRIME))
        mu = mobius(num)
        mu_sq = 1.0 if (mu != 0) else 0.0
        gauss = math.exp(-0.5 * ((float(j) - float(j_target)) / sigma) ** 2)
        psi[j] = chi * mu_sq * gauss

    n = float(np.linalg.norm(psi))
    if n < 1e-15:
        # Fallback: Gaussian bez karakter-filtera (održava lokalnost oko j_target)
        for j in range(DIM):
            num = position_1based + j
            if num < 1 or num > N_MAX or num <= prev_pick:
                continue
            psi[j] = math.exp(-0.5 * ((float(j) - float(j_target)) / sigma) ** 2)
        n = float(np.linalg.norm(psi))
        if n < 1e-15:
            return psi
    return psi / n


# =========================
# Predikcija jedne pozicije
# =========================
def nt_pick_one_position(
    position_1based: int,
    prev_pick: int,
    rng: np.random.Generator,
) -> Tuple[int, int, float, float, float, float, float, float]:
    j_target, target = compute_j_target(position_1based, prev_pick)

    psi_0 = build_char_state(j_target, position_1based, prev_pick, SIGMA_INIT)
    psi_qw = U_KIN @ psi_0
    n = float(np.linalg.norm(psi_qw))
    if n < 1e-15:
        psi_fin = psi_0
    else:
        psi_fin = psi_qw / n

    probs = np.abs(psi_fin) ** 2
    probs = np.clip(np.real(probs), 0.0, None)

    js = np.arange(DIM, dtype=np.float64)
    mean_j = float(np.sum(js * probs))
    var_j = float(np.sum(((js - mean_j) ** 2) * probs))

    qr_mass = 0.0
    sqf_mass = 0.0
    for j in range(DIM):
        num = position_1based + j
        if 1 <= num <= N_MAX:
            if (num % P_PRIME) in QR_SET:
                qr_mass += probs[j]
            if mobius(num) != 0:
                sqf_mass += probs[j]

    # Shannon entropy
    eps = 1e-15
    entropy = -float(np.sum(probs * np.log(probs + eps)))

    mask = np.zeros(DIM, dtype=np.float64)
    for j in range(POS_RANGE):
        num = position_1based + j
        if 1 <= num <= N_MAX and num > prev_pick:
            mask[j] = 1.0

    probs_valid = probs * mask
    s = float(probs_valid.sum())
    if s < 1e-15:
        for j in range(POS_RANGE):
            num = position_1based + j
            if 1 <= num <= N_MAX and num > prev_pick:
                return (
                    num, j_target, target, mean_j, var_j,
                    qr_mass, sqf_mass, entropy,
                )
        return (
            max(prev_pick + 1, position_1based),
            j_target, target, mean_j, var_j,
            qr_mass, sqf_mass, entropy,
        )

    probs_valid /= s
    j_sampled = int(rng.choice(DIM, p=probs_valid))
    num = position_1based + j_sampled
    return num, j_target, target, mean_j, var_j, qr_mass, sqf_mass, entropy


# =========================
# Autoregresivni run
# =========================
def run_nt_autoregressive() -> List[int]:
    rng = np.random.default_rng(SEED)
    picks: List[int] = []
    prev_pick = 0

    for i in range(1, N_NUMBERS + 1):
        (num, j_t, target, mean_j, var_j,
         qr_mass, sqf_mass, entropy) = nt_pick_one_position(
            i, prev_pick, rng
        )
        picks.append(int(num))
        print(
            f"  [pos {i}]  target={target:.3f}  j_target={j_t:2d}  "
            f"⟨j⟩={mean_j:5.2f}  σ_j={math.sqrt(max(var_j,0)):.3f}  "
            f"QR={qr_mass:.3f}  SQF={sqf_mass:.3f}  H={entropy:.3f}  "
            f"num={num:2d}"
        )
        prev_pick = int(num)

    return picks


# =========================
# Main
# =========================
def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Nema CSV: {CSV_PATH}")

    H = load_rows(CSV_PATH)
    H_sorted = sort_rows_asc(H)
    S_bar = float(H_sorted.sum(axis=1).mean())

    qr_in_range = sorted([n for n in range(1, N_MAX + 1) if (n % P_PRIME) in QR_SET])
    sqf_in_range = sorted([n for n in range(1, N_MAX + 1) if mobius(n) != 0])

    print("=" * 88)
    print("Q42 Teorija brojeva — Dirichlet karakter χ_37 + Möbius μ + Chevalley šetnja")
    print("=" * 88)
    print(f"CSV:            {CSV_PATH}")
    print(f"Broj redova:    {H.shape[0]}")
    print(f"Qubit budget:   {NQ} po poziciji  (Hilbert dim={DIM})")
    print(f"Prost p:        {P_PRIME}  (Legendreov simbol χ_p(n) = (n/p))")
    print(f"QR mod 37 u [1,39]:  {qr_in_range}  ({len(qr_in_range)} brojeva)")
    print(f"SQF u [1,39]:         {sqf_in_range}  ({len(sqf_in_range)} brojeva)")
    print(f"Inicijalno st.: |ψ_0⟩ ∝ Σ_j χ_p(num) · μ²(num) · g_σ(j − j_target) |j⟩")
    print(f"Parametri:      σ = {SIGMA_INIT}  (Gaussian lok.)   J = {J_HOP}   t* = {T_STAR}")
    print(f"Dinamika:       |ψ_QW⟩ = exp(−i·H_kin·t*) · |ψ_0⟩  (SU(64) kvantna šetnja)")
    print(f"Srednja suma S̄: {S_bar:.3f}  (CSV info, nije driver)")
    print(f"Seed:           {SEED}")
    print()
    print("Pokretanje teorije brojeva (χ + μ² + Chevalley walk) po pozicijama:")

    picks = run_nt_autoregressive()

    n_odd = sum(1 for v in picks if v % 2 == 1)
    gaps = [picks[i + 1] - picks[i] for i in range(N_NUMBERS - 1)]
    n_qr = sum(1 for v in picks if (v % P_PRIME) in QR_SET)
    n_sqf = sum(1 for v in picks if mobius(v) != 0)

    print()
    print("=" * 88)
    print("REZULTAT Q42 (NEXT kombinacija)")
    print("=" * 88)
    print(f"Suma:   {sum(picks)}   (S̄={S_bar:.2f})")
    print(f"#odd:   {n_odd}")
    print(f"#QR:    {n_qr} / 7   (mod {P_PRIME})")
    print(f"#SQF:   {n_sqf} / 7   (squarefree)")
    print(f"Gaps:   {gaps}")
    print(f"Predikcija NEXT: {picks}")


if __name__ == "__main__":
    main()



"""
========================================================================================
Q42 Teorija brojeva — Dirichlet karakter χ_37 + Möbius μ + Chevalley šetnja
========================================================================================
CSV:            /data/loto7hh_4602_k32.csv
Broj redova:    4602
Qubit budget:   6 po poziciji  (Hilbert dim=64)
Prost p:        37  (Legendreov simbol χ_p(n) = (n/p))
QR mod 37 u [1,39]:  [1, 3, 4, 7, 9, 10, 11, 12, 16, 21, 25, 26, 27, 28, 30, 33, 34, 36, 38]  (19 brojeva)
SQF u [1,39]:         [1, 2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19, 21, 22, 23, 26, 29, 30, 31, 33, 34, 35, 37, 38, 39]  (26 brojeva)
Inicijalno st.: |ψ_0⟩ ∝ Σ_j χ_p(num) · μ²(num) · g_σ(j − j_target) |j⟩
Parametri:      σ = 2.5  (Gaussian lok.)   J = 1.0   t* = 1.0
Dinamika:       |ψ_QW⟩ = exp(−i·H_kin·t*) · |ψ_0⟩  (SU(64) kvantna šetnja)
Srednja suma S̄: 140.509  (CSV info, nije driver)
Seed:           39

Pokretanje teorije brojeva (χ + μ² + Chevalley walk) po pozicijama:
  [pos 1]  target=4.875  j_target= 4  ⟨j⟩= 4.01  σ_j=2.309  QR=0.444  SQF=0.740  H=2.036  num= 5
  [pos 2]  target=9.857  j_target= 8  ⟨j⟩= 8.38  σ_j=2.321  QR=0.702  SQF=0.644  H=2.134  num=11
  [pos 3]  target=15.667  j_target=13  ⟨j⟩=12.65  σ_j=1.956  QR=0.406  SQF=0.427  H=1.847  num=14
  [pos 4]  target=19.000  j_target=15  ⟨j⟩=15.29  σ_j=2.352  QR=0.265  SQF=0.403  H=1.779  num=18
  [pos 5]  target=23.250  j_target=18  ⟨j⟩=17.45  σ_j=2.187  QR=0.406  SQF=0.551  H=2.019  num=23
  [pos 6]  target=28.333  j_target=22  ⟨j⟩=22.77  σ_j=2.156  QR=0.781  SQF=0.531  H=1.967  num=30
  [pos 7]  target=34.500  j_target=27  ⟨j⟩=26.90  σ_j=2.161  QR=0.554  SQF=0.704  H=1.977  num=32

========================================================================================
REZULTAT Q42 (NEXT kombinacija)
========================================================================================
Suma:   133   (S̄=140.51)
#odd:   3
#QR:    2 / 7   (mod 37)
#SQF:   5 / 7   (squarefree)
Gaps:   [6, 3, 4, 5, 7, 2]
Predikcija NEXT: [5, 11, x, y, z, 30, 32]
"""



"""
REZULTAT — Q42 Teorija brojeva / Dirichlet karakter + Möbius + kvantna šetnja
-----------------------------------------------------------------------------
(Popunjava se iz printa main()-a nakon pokretanja.)

Koncept:
  • Čisto kvantno: state preparation sa character-signed amplitude,
    SU(64) Chevalley kvantna šetnja, Born sempling. Bez klasičnog ML-a.
  • Teorija brojeva: Legendreov simbol χ_37 (QR mod 37), Möbius funkcija μ
    (squarefree filter), Eulerov kriterijum za izračunavanje χ_p preko
    modularnog stepenovanja.
  • Granica sa period-finding: koristi iste multiplikativne objekte (prost p,
    karakteri) KAO Shor/Simon, ali ne ekstraktuje period. Static character-
    stanje + Lie Chevalley dinamika.
  • ne-period-finding paradigma, različita od Q28/Q29/Q30 (QFR,
    QSVE, QSP) i Q31 (Simon XOR-period).
  • p = 37 prirodno se uklapa u opseg [1, 39]; karakter i Möbius
    daju aritmetičke invarijante koje nisu frekvencijske.
  • NQ = 6 qubit-a po poziciji, reciklirani 64-dim registar.
  • deterministički seed + fiksni σ, J, t* + seeded Born sempling.

Tehnike:
  • Eulerov kriterijum: χ_p(n) = n^{(p−1)/2} mod p  (∈ {0, 1, −1 ≡ p−1}).
  • Möbius preko direktne faktorizacije do sqrt(n).
  • Gaussian Lokalnost: g_σ(j − j_target) = exp(−(j − j_target)²/(2σ²)).
  • SU(64) Chevalley H_kin = −J(T̂_+ + T̂_−), evolucija preko eigh.
  • Born sempling iz maskovane distribucije |⟨j|ψ_QW⟩|².

Dijagnostike:
  • QR mass = suma verovatnoća na kvadratnim ostatcima mod 37.
  • SQF mass = suma verovatnoća na squarefree brojevima.
  • Shannon entropy H(P) raspodele.
"""
