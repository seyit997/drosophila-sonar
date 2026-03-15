import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import streamlit as st

# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Drosophila Sonar Sim",
    page_icon="🦟",
    layout="wide"
)

st.markdown("""
<style>
body { background-color: #0a0e1a; color: #e2e8f0; }
.block-container { background-color: #0a0e1a; }
h1, h2, h3 { color: #00d4ff; font-family: monospace; }
</style>
""", unsafe_allow_html=True)

st.title("🦟 Drosophila Melanogaster — Sonar Cross-Modal Simülasyon")
st.markdown("""
**FlyWire Connectome** tabanlı meyve sineği beyin devresi.  
Yarasa sonarına benzer ultrasonik pulse → Johnston's Organ → AMMC → Giant Fiber → Motor çıkış.
""")

# ══════════════════════════════════════════════════════════════════════
# KENAR ÇUBUĞU — kullanıcı parametreleri
# ══════════════════════════════════════════════════════════════════════
st.sidebar.header("⚙️ Simülasyon Parametreleri")

pulse_count     = st.sidebar.slider("Echo sayısı",          2, 8,  5)
echo_start      = st.sidebar.slider("İlk echo zamanı (ms)", 5, 30, 10)
echo_spacing    = st.sidebar.slider("Echo aralığı (ms)",    8, 30, 18)
approach        = st.sidebar.radio("Nesne hareketi",
                                   ["Yaklaşıyor", "Uzaklaşıyor"]) == "Yaklaşıyor"
amplitude       = st.sidebar.slider("Stimulus gücü",        0.1, 2.0, 1.0, 0.1)
sim_duration    = st.sidebar.slider("Simülasyon süresi (ms)", 80, 250, 150)
noise_level     = st.sidebar.slider("Biyolojik gürültü",    0.0, 0.5, 0.15, 0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Devre Yolu:**
`JO → AMMC → GF → DLM` (kaçış)  
`AMMC → WED → MB → LH` (yenilik)
""")

run_btn = st.sidebar.button("▶ Simülasyonu Çalıştır", type="primary")

# ══════════════════════════════════════════════════════════════════════
# HODGKIN-HUXLEY BENZERİ LIF NÖRON
# ══════════════════════════════════════════════════════════════════════

class LIFNeuron:
    """
    Leaky Integrate-and-Fire nöron.
    Drosophila internöron parametreleri (Günay et al. 2008).
    τ * dv/dt = (v_rest - v) + R * I
    """
    def __init__(self, tau=10.0, v_rest=-65.0, v_thresh=-50.0,
                 v_reset=-70.0, R=200.0, t_ref=2.0, ntype="interneuron"):
        self.tau      = tau
        self.v_rest   = v_rest
        self.v_thresh = v_thresh
        self.v_reset  = v_reset
        self.R        = R
        self.t_ref    = t_ref
        self.ntype    = ntype
        self.reset_state()

    def reset_state(self):
        self.v           = self.v_rest
        self.ref_timer   = 0.0
        self.spike_times = []
        self.v_trace     = []

    def step(self, I_ext, dt, t_now, noise=0.15):
        self.v_trace.append(self.v)
        if self.ref_timer > 0:
            self.ref_timer -= dt
            return False

        I_noise = np.random.normal(0, noise)
        dv = (self.v_rest - self.v + self.R * (I_ext + I_noise)) / self.tau
        self.v += dv * dt

        if self.v >= self.v_thresh:
            self.v = self.v_reset
            self.ref_timer = self.t_ref
            self.spike_times.append(t_now)
            return True
        return False


# ══════════════════════════════════════════════════════════════════════
# DEVRE TANIMI
# ══════════════════════════════════════════════════════════════════════

def build_circuit():
    # Nöron tipleri ve parametreleri
    circuit = {
        "JO-A":  LIFNeuron(tau=8,  v_rest=-65, v_thresh=-50, R=180, ntype="mechanosensory"),
        "JO-B":  LIFNeuron(tau=8,  v_rest=-65, v_thresh=-50, R=190, ntype="mechanosensory"),
        "AMMC":  LIFNeuron(tau=10, v_rest=-65, v_thresh=-51, R=200, ntype="interneuron"),
        "WED":   LIFNeuron(tau=12, v_rest=-66, v_thresh=-51, R=210, ntype="interneuron"),
        "GF":    LIFNeuron(tau=6,  v_rest=-63, v_thresh=-48, R=220, ntype="giant_fiber"),
        "MB":    LIFNeuron(tau=15, v_rest=-70, v_thresh=-52, R=180, ntype="mushroom_body"),
        "LH":    LIFNeuron(tau=11, v_rest=-66, v_thresh=-51, R=195, ntype="interneuron"),
        "DLM":   LIFNeuron(tau=7,  v_rest=-64, v_thresh=-49, R=215, ntype="motor"),
    }
    return circuit

# Sinaptik ağırlıklar (FlyWire synapse yoğunluğundan normalize)
SYNAPSES = {
    # (pre, post): (ağırlık, gecikme_ms)
    ("JO-A",  "AMMC"): (0.55, 0.5),
    ("JO-B",  "AMMC"): (0.80, 0.4),   # yüksek frekans → güçlü bağlantı
    ("JO-B",  "WED"):  (0.45, 0.7),
    ("AMMC",  "GF"):   (0.90, 0.2),   # hızlı kaçış yolu
    ("AMMC",  "WED"):  (0.40, 0.6),
    ("WED",   "MB"):   (0.35, 1.2),
    ("MB",    "LH"):   (0.40, 0.8),
    ("GF",    "DLM"):  (0.95, 0.15),  # tam bağlantı — kaçış motoru
}


# ══════════════════════════════════════════════════════════════════════
# SONAR PULSE ÜRETİCİ
# ══════════════════════════════════════════════════════════════════════

def make_sonar_signal(t_arr, pulse_count, echo_start, echo_spacing,
                      approach, amplitude):
    """
    FM sweep sonar pulse'ları üret.
    Gaussian zarf × sinüs sweep = gerçekçi sonar pulse.
    """
    signal = np.zeros(len(t_arr))
    echo_times = []

    for p in range(pulse_count):
        if approach:
            t_e = echo_start + echo_spacing * p * max(0.3, 1.0 - 0.12 * p)
        else:
            t_e = echo_start + echo_spacing * p * (1.0 + 0.08 * p)

        if t_e > t_arr[-1]:
            break
        echo_times.append(t_e)

        width = 3.0
        mask  = (t_arr >= t_e) & (t_arr <= t_e + width)
        if mask.sum() == 0:
            continue

        tau      = t_arr[mask] - t_e
        envelope = np.exp(-2.5 * tau)
        sweep    = np.sin(2 * np.pi * (0.8 - 0.3 * tau / width) * tau * 20)
        decay    = max(0.3, 1.0 - p * 0.08)
        signal[mask] += amplitude * decay * envelope * sweep

    return signal, echo_times


# ══════════════════════════════════════════════════════════════════════
# ANA SİMÜLASYON
# ══════════════════════════════════════════════════════════════════════

def run_simulation(pulse_count, echo_start, echo_spacing, approach,
                   amplitude, sim_duration, noise_level):

    dt    = 0.05   # ms
    t_arr = np.arange(0, sim_duration, dt)

    # Sonar sinyali üret
    sonar_signal, echo_times = make_sonar_signal(
        t_arr, pulse_count, echo_start, echo_spacing, approach, amplitude
    )

    # Devre kur
    circuit = build_circuit()
    for n in circuit.values():
        n.reset_state()

    # Sinaptik gecikme tamponları
    delay_buffers = {}
    for (pre, post), (w, delay_ms) in SYNAPSES.items():
        buf_len = max(1, int(delay_ms / dt))
        delay_buffers[(pre, post)] = {
            "buffer": np.zeros(buf_len),
            "weight": w,
            "ptr":    0,
        }

    # Zaman döngüsü
    I_syn_current = {name: 0.0 for name in circuit}

    for step_i, t in enumerate(t_arr):

        # JO-A ve JO-B'ye sonar akımı ver
        jo_gain = {
            "JO-A": 3.5,   # düşük-orta frekans
            "JO-B": 8.0,   # yüksek frekans — sonar buraya
        }
        I_ext = {
            "JO-A":  abs(sonar_signal[step_i]) * jo_gain["JO-A"],
            "JO-B":  abs(sonar_signal[step_i]) * jo_gain["JO-B"],
            "AMMC":  I_syn_current["AMMC"],
            "WED":   I_syn_current["WED"],
            "GF":    I_syn_current["GF"],
            "MB":    I_syn_current["MB"],
            "LH":    I_syn_current["LH"],
            "DLM":   I_syn_current["DLM"],
        }

        # Nöronları güncelle
        spikes_this_step = {}
        for name, neuron in circuit.items():
            fired = neuron.step(I_ext[name], dt, t, noise=noise_level)
            spikes_this_step[name] = fired

        # Sinaptik akımları güncelle (gecikmeli)
        I_syn_next = {name: 0.0 for name in circuit}
        for (pre, post), buf_data in delay_buffers.items():
            buf  = buf_data["buffer"]
            w    = buf_data["weight"]
            ptr  = buf_data["ptr"]
            buf_len = len(buf)

            # Spike varsa tampona yaz
            if spikes_this_step.get(pre, False):
                buf[ptr] = w * 0.8   # nA cinsinden eşdeğer

            # En eski değeri oku (gecikme kadar önce yazılan)
            read_idx = (ptr + 1) % buf_len
            I_syn_next[post] += buf[read_idx]
            buf[read_idx] = 0.0

            buf_data["ptr"] = (ptr + 1) % buf_len

        I_syn_current = I_syn_next

    # Sonuçları topla
    results = {}
    for name, neuron in circuit.items():
        results[name] = {
            "spike_times": neuron.spike_times,
            "spike_count": len(neuron.spike_times),
            "v_trace":     np.array(neuron.v_trace),
        }

    return t_arr, sonar_signal, echo_times, results


# ══════════════════════════════════════════════════════════════════════
# DAVRANIŞSAL YORUM
# ══════════════════════════════════════════════════════════════════════

def compute_behavior(results):
    sc    = {name: results[name]["spike_count"] for name in results}
    total = float(sum(sc.values())) + 1e-9

    escape  = float(sc["GF"] * 3 + sc["DLM"] * 2) / (total * 0.4 + 1.0)
    novelty = float(sc["MB"] * 2 + sc["WED"])      / (total * 0.3 + 1.0)
    freeze  = float(max(0.0, 1.2 - escape - novelty * 0.6))

    raw = np.array([escape, novelty, freeze], dtype=float)
    if raw.sum() > 0:
        raw /= raw.sum()

    labels   = ["Kaçış", "Yenilik/Merak", "Donma"]
    dominant = labels[int(np.argmax(raw))]
    emojis   = ["🏃", "🔍", "🧊"]
    dom_emoji = emojis[int(np.argmax(raw))]

    return dict(zip(labels, raw.tolist())), dominant, dom_emoji


# ══════════════════════════════════════════════════════════════════════
# GÖRSELLEŞTİRME
# ══════════════════════════════════════════════════════════════════════

def make_figure(t_arr, sonar_signal, echo_times, results,
                behavior_scores, dominant, dom_emoji, sim_duration):

    BG     = '#0a0e1a'
    PANEL  = '#111827'
    GR     = '#1e2d3d'
    TX     = '#e2e8f0'
    COLS   = {
        'JO-A': '#00d4ff', 'JO-B': '#38bdf8',
        'AMMC': '#60a5fa', 'WED':  '#a78bfa',
        'GF':   '#f97316', 'MB':   '#c084fc',
        'LH':   '#fb7185', 'DLM':  '#4ade80',
    }

    fig = plt.figure(figsize=(18, 12), facecolor=BG)
    fig.suptitle(
        "DROSOPHILA MELANOGASTER — SONAR CROSS-MODAL SİMÜLASYONU",
        color='#00d4ff', fontsize=14, fontweight='bold',
        fontfamily='monospace', y=0.98
    )
    gs = gridspec.GridSpec(3, 3, figure=fig,
                           hspace=0.55, wspace=0.38,
                           top=0.94, bottom=0.06,
                           left=0.07, right=0.97)

    order = ['JO-A', 'JO-B', 'AMMC', 'WED', 'GF', 'MB', 'LH', 'DLM']

    # ── [A] Spike Raster ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, :2])
    ax.set_facecolor(PANEL)
    y_off, yticks, ylabels = 0, [], []
    for name in order:
        sp_t = results[name]["spike_times"]
        ax.scatter(sp_t, [y_off + 0.5] * len(sp_t),
                   s=12, c=COLS[name], alpha=0.9, linewidths=0, marker='|')
        yticks.append(y_off + 0.5)
        ylabels.append(f'{name}  ({results[name]["spike_count"]})')
        ax.axhline(y_off + 1, color=GR, lw=0.4)
        y_off += 1

    for t_e in echo_times:
        ax.axvline(float(t_e), color='#fbbf24', alpha=0.35, lw=1.0, ls='--')

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, color=TX, fontsize=7, fontfamily='monospace')
    ax.set_xlabel('Zaman (ms)', color=TX, fontsize=8)
    ax.set_title('Spike Raster — Sensörden Motora Yayılım',
                 color=TX, fontsize=9, fontfamily='monospace')
    ax.tick_params(colors=TX, labelsize=7)
    ax.set_xlim(0, sim_duration)
    for s in ax.spines.values(): s.set_color(GR)
    ax.grid(True, color=GR, alpha=0.3, lw=0.3, axis='x')

    # ── [B] GF Voltaj ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor(PANEL)
    v = results['GF']['v_trace']
    t_plot = np.linspace(0, sim_duration, len(v))
    ax2.plot(t_plot, v, color=COLS['GF'], lw=1.0)
    ax2.axhline(-50, color='white', alpha=0.2, lw=0.5, ls='--')
    ax2.text(1, -49, 'eşik', color='white', fontsize=6,
             fontfamily='monospace', alpha=0.5)
    ax2.set_title('Giant Fiber Voltajı\n(kaçış nöronu)',
                  color=COLS['GF'], fontsize=8, fontfamily='monospace')
    ax2.set_xlabel('ms', color=TX, fontsize=7)
    ax2.set_ylabel('mV', color=TX, fontsize=7)
    ax2.tick_params(colors=TX, labelsize=6)
    for s in ax2.spines.values(): s.set_color(GR)
    ax2.grid(True, color=GR, alpha=0.3, lw=0.3)

    # ── [C] Spike Bar ────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.set_facecolor(PANEL)
    counts = [results[n]["spike_count"] for n in order]
    cols_b = [COLS[n] for n in order]
    bars   = ax3.bar(order, counts, color=cols_b, alpha=0.85, width=0.6)
    for bar, cnt in zip(bars, counts):
        ax3.text(bar.get_x() + bar.get_width()/2, cnt + 0.1,
                 str(cnt), ha='center', color=TX,
                 fontsize=8, fontfamily='monospace', fontweight='bold')
    ax3.set_title('Bölge Başına Spike Sayısı',
                  color=TX, fontsize=9, fontfamily='monospace')
    ax3.set_ylabel('# Spike', color=TX, fontsize=8)
    ax3.tick_params(colors=TX, labelsize=8)
    for s in ax3.spines.values(): s.set_color(GR)
    ax3.grid(True, color=GR, alpha=0.3, lw=0.3, axis='y')

    # ── [D] Davranış ─────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.set_facecolor(PANEL)
    b_labels = list(behavior_scores.keys())
    b_vals   = list(behavior_scores.values())
    b_cols   = ['#f97316', '#c084fc', '#38bdf8']
    b_bars   = ax4.bar(b_labels, b_vals, color=b_cols, alpha=0.85, width=0.5)
    for bar, val in zip(b_bars, b_vals):
        ax4.text(bar.get_x() + bar.get_width()/2, float(val) + 0.02,
                 f'{float(val):.2f}', ha='center', color=TX,
                 fontsize=9, fontfamily='monospace', fontweight='bold')
    ax4.set_ylim(0, 1.15)
    ax4.set_title(f'Davranış  {dom_emoji} {dominant}',
                  color='#fbbf24', fontsize=10,
                  fontfamily='monospace', fontweight='bold')
    ax4.tick_params(colors=TX, labelsize=8)
    for s in ax4.spines.values(): s.set_color(GR)
    ax4.grid(True, color=GR, alpha=0.3, lw=0.3, axis='y')

    # ── [E] Sonar Echo ───────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, :])
    ax5.set_facecolor(PANEL)
    ax5.plot(t_arr, sonar_signal, color='#00d4ff', lw=1.0, alpha=0.9)
    ax5.fill_between(t_arr, sonar_signal, alpha=0.1, color='#00d4ff')

    for i, t_e in enumerate(echo_times):
        ax5.axvline(float(t_e), color='#fbbf24', alpha=0.45, lw=1.0, ls='--')
        if float(t_e) < sim_duration - 5:
            ax5.text(float(t_e) + 0.3, ax5.get_ylim()[1] * 0.75,
                    f'E{i+1}', color='#fbbf24', fontsize=7,
                    fontfamily='monospace')

    for i in range(len(echo_times) - 1):
        gap   = float(echo_times[i+1]) - float(echo_times[i])
        x_mid = (float(echo_times[i]) + float(echo_times[i+1])) / 2.0
        y_arr = float(ax5.get_ylim()[0]) * 0.7
        ax5.annotate('', xy=(float(echo_times[i+1]), y_arr),
                    xytext=(float(echo_times[i]), y_arr),
                    arrowprops=dict(arrowstyle='<->', color='#a78bfa', lw=1.0))
        ax5.text(x_mid, y_arr * 1.15, f'{gap:.1f}ms', ha='center',
                color='#a78bfa', fontsize=6, fontfamily='monospace')

    direction = "← YAKINLAŞIYOR" if approach else "→ UZAKLAŞIYOR"
    ax5.set_title(
        f'Sonar Echo Paterni — {direction}',
        color=TX, fontsize=9, fontfamily='monospace')
    ax5.set_xlabel('Zaman (ms)', color=TX, fontsize=8)
    ax5.set_ylabel('Amplitüd',   color=TX, fontsize=8)
    ax5.set_xlim(0, sim_duration)
    ax5.tick_params(colors=TX, labelsize=7)
    for s in ax5.spines.values(): s.set_color(GR)
    ax5.grid(True, color=GR, alpha=0.3, lw=0.3)

    plt.close('all')
    return fig


# ══════════════════════════════════════════════════════════════════════
# STREAMLIT ANA AKIŞ
# ══════════════════════════════════════════════════════════════════════

if run_btn:
    with st.spinner("Simülasyon çalışıyor..."):
        np.random.seed(42)
        t_arr, sonar_sig, echo_t, results = run_simulation(
            pulse_count, echo_start, echo_spacing, approach,
            amplitude, sim_duration, noise_level
        )
        beh_scores, dominant, dom_emoji = compute_behavior(results)

    # Metrik kartları
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Giant Fiber Spike",  results["GF"]["spike_count"],  "kaçış nöronu")
    col2.metric("Motor (DLM) Spike",  results["DLM"]["spike_count"], "kanat kası")
    col3.metric("Mushroom Body",      results["MB"]["spike_count"],  "yenilik")
    col4.metric(f"Baskın: {dom_emoji}", dominant)

    st.markdown("---")

    # Ana grafik
    fig = make_figure(t_arr, sonar_sig, echo_t, results,
                      beh_scores, dominant, dom_emoji, sim_duration)
    st.pyplot(fig)

    # Detay tablosu
    st.markdown("### 📊 Spike Sayımı Detayı")
    cols = st.columns(len(results))
    for i, (name, data) in enumerate(results.items()):
        cols[i].metric(name, data["spike_count"], "spike")

    # Biyolojik yorum
    st.markdown("### 🧬 Biyolojik Yorum")
    if dominant == "Kaçış":
        st.success("""
        **Giant Fiber (GF) aktive oldu → DLM motor nöronu ateşlendi.**  
        Bu Drosophila'nın klasik kaçış refleksi: GF→DLM yolu 1ms'den kısa sürede 
        kanat kasını kasarak uçuşu başlatır. Sonar pulse yüksek frekanslı olduğu için 
        JO-B → AMMC → GF yolu tercihli aktive edildi.
        """)
    elif dominant == "Yenilik/Merak":
        st.info("""
        **Mushroom Body (MB) ve Lateral Horn (LH) ön planda.**  
        GF aktivasyonu düşük kaldı — sinyal kaçış eşiğini geçemedi.  
        WED → MB yolu aktive oldu: bu Drosophila'nın alışılmamış bir uyarıyı 
        "kaydettiğini" ve değerlendirdiğini gösterir. Öğrenme bağlamında kritik devre.
        """)
    else:
        st.warning("""
        **Düşük genel aktivite — donma/bekleme.**  
        Uyarı ne kaçış eşiğini ne de yenilik devresini tam aktive edebildi.  
        Drosophila'nın alışık olmadığı ama tehdit olarak algılamadığı bir uyarıya 
        verdiği pasif tepkiyi yansıtıyor.
        """)

else:
    st.info("👈 Sol panelden parametreleri ayarla, ardından **▶ Simülasyonu Çalıştır** butonuna bas.")

    st.markdown("""
    ### 🔬 Bu simülasyon ne yapıyor?

    | Bölge | Biyolojik Karşılık | Fonksiyon |
    |-------|-------------------|-----------|
    | **JO-A / JO-B** | Johnston's Organ | Anten mekanosensörleri — ses/titreşim |
    | **AMMC** | Antennal Mech. Motor Center | İşitme entegrasyon merkezi |
    | **WED** | Wedge | Yenilik tespiti upstream |
    | **GF** | Giant Fiber | Kaçış refleks nöronu |
    | **MB** | Mushroom Body | Öğrenme / bellek |
    | **LH** | Lateral Horn | Doğuştan tepki |
    | **DLM** | Dorsal Long. Muscle motor | Kanat kası — uçuş |

    **Kaynak:** Zheng et al. 2018 (Janelia hemibrain), Schlegel et al. 2023 (FlyWire full connectome)
    """)
    