import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
from scipy.signal import welch
import plotly.graph_objects as go
import math

# Page Configuration
st.set_page_config(page_title="HRV Analysis & RR Tachogram Detection", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .cta-button {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 30px;
    }
    .background {
        background-image: url('Bedah Jantung.jpg');
        background-size: cover;
        padding: 50px 0;
        text-align: center;
        color: white;
    }
    .background h1 {
        margin: 0;
        font-size: 2.5em;
    }
    </style>
    """, unsafe_allow_html=True)

# Navigation Function
def switch_page(page):
    if page == "Dashboard":
        st.session_state.page = "Dashboard"
    else:
        st.session_state.page = "Landing"

# Landing Page Sections
def header_section():
    st.markdown("<h1 style='text-align: center;color: #F7F8F7;'>Your heart's rhythm reveals more than just beats</h1> <h4 style='text-align: center;color: #555;'>Unlock the SECRETS of your heart's rhythm </h4>", unsafe_allow_html=True)

def user_concerns_section():
    st.markdown("<h2 style='text-align: center; color: #F7F8F7;'>Unseen Dangers Awaiting You</h2>", unsafe_allow_html=True)
    st.image('Orang Sakit.png', use_column_width=True) 
    st.markdown("<h4 style='text-align: center; color: #EB455F;'>Faktanya: lebih dari 17.9 juta orang meninggal tiap tahunnya akibat penyakit jantung.</h4>", unsafe_allow_html=True)
    st.write("""
        Kita terlalu sibuk, sampai kita lupa dan abai terhadap kesehatan jantung kita. Tanpa kita sadari ada banyak ancaman yang mendekat:
         - Peningkatan risiko penyakit kardiovaskular.
         - Tingkat stres tanpa disadari yang memengaruhi kesejahteraan secara keseluruhan.
         - Hilangnya peringatan dini mengenai potensi kondisi jantung.
    """)

def causes_section():
    st.markdown("<h2 class='header'>Recognize it early</h2>", unsafe_allow_html=True)
    st.write("""
        Sadar akan pentingnya deteksi HRV akan membantu kamu terhindar dari masalah yang lebih besar, seperti:
          1. **Stres Kronis**: Stres yang berkepanjangan dapat mengganggu ritme jantung, sehingga menyebabkan masalah kesehatan jangka panjang.
          2. **Pilihan Gaya Hidup yang Buruk**: Kebiasaan makan yang tidak sehat, kurang olahraga, dan kurang tidur berkontribusi signifikan terhadap masalah jantung.
          3. **Kurangnya Kesadaran**: Banyak orang tidak menyadari pentingnya memantau kesehatan jantung mereka secara teratur.
    """)

def testimonials_section():
    st.markdown("<h2 class='header'>Hear from Our Community</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image('Ardi.png', width=150)  # Replace with your actual image file
        st.markdown("<h4 class='testimonial'>Rahardian Asyam Zuhdi</h4>", unsafe_allow_html=True)
        st.write("""
            Setelah mempelajari analisis HRV,  stress saya menjadi berkurang. Saya bisa berolahraga dengan teratur dan berkeliling dunia kapanpun saya mau.
        """)
        
    with col2:
        st.image('Bahari.png', width=150)  # Replace with your actual image file
        st.markdown("<h4 class='testimonial'>Bahari Noor Hidayat</h4>", unsafe_allow_html=True)
        st.write("""
            Deteksi RR Tachogram membantu saya mengetahui kondisi kesehatan saya sejak dini, saya tidak lagi merasa khawatir menentukan penanganan yang tepat untuk rasa sakit yang saya alami.
        """)
        
    with col3:
        st.image('Qusay.png', width=150)  # Replace with your actual image file
        st.markdown("<h4 class='testimonial'>Muhammad Qusay Yubasyrendra Akib</h4>", unsafe_allow_html=True)
        st.write("""
            Memahami HRV berarti mulai memahami diri sendiri. Saya menjadi lebih mampu untuk mengelola mengelola kecemasan saya, akhirnya kualitas tidur saya menjadi lebih baik.
        """)

def cta_section():
    st.markdown("<h2 style='text-align: center; color: #4CAF50;'>It's Never Too Late</h2>", unsafe_allow_html=True)
    st.write("<h5 style='text-align: center;color: #F7F8F7;'>Kami percaya setiap orang ingin memberikan yang terbaik untuk dirinya dan orang di sekitranya</h2>", unsafe_allow_html=True)
    st.markdown("<div class='cta-button'><a href='#' onclick='window.location.reload()' style='background-color: #4CAF50; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;'>I'm ready to be more healthy</a></div>", unsafe_allow_html=True)
    if st.button("I'm Aware"):
        switch_page("Dashboard")

def landing_page():
    header_section()
    st.markdown("---")
    user_concerns_section()
    st.markdown("---")
    causes_section()
    st.markdown("---")
    testimonials_section()
    st.markdown("---")
    cta_section()

############ DASHBOARD PAGE
#Upload Dataset 
def plot_data():  
    data_file = st.file_uploader("Upload Dataset", type=["txt","xlsx"])
    if data_file is not None:
        try:
            column_names = ['ECG']
            if data_file.name.endswith('.txt'):
                try:
                    data = pd.read_csv(data_file, delimiter="\t", header=None, encoding='utf-8',names=column_names)
                except UnicodeDecodeError:
                    data = pd.read_csv(data_file, delimiter="\t", header=None, encoding='ISO-8859-1',names=column_names)
            elif data_file.name.endswith('.xlsx'):
                data = pd.read_excel(data_file, header=None,names=column_names)            #data = pd.read_csv(data_file, sep="\t",header=None)  # Pastikan delimiter sesuai dengan file txt
            # Convert the ECG column to numeric (if there are any non-numeric values, they will be coerced to NaN)
            data['ECG'] = pd.to_numeric(data['ECG'], errors='coerce')

            # Drop any rows with NaN values that may have been introduced due to conversion issues
            data.dropna(inplace=True)

            # Create sample interval and elapsed time columns
            N = len(data)
            # Define x and y
            x = np.zeros(N)
            y = np.zeros(N)
            x = np.arange(len(data)) * (1/200)
            y = data["ECG"] - data["ECG"].mean()
            st.dataframe(data)
            st.write("Visualisasi dataset")
            fig, ax = plt.subplots(figsize=(14, 10))
            ax.plot(data.index, data, label='Data', color='blue')
            ax.set_xlabel('Time')
            ax.set_ylabel('Amplitude')
            ax.set_title('ECG Data Visualization')
            plt.tight_layout()
            st.pyplot(fig)
            #plot 24400 data
            st.write("Grafik sinyal ECG dari dataset")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(data.index[:5000], data[:5000], label='First 5000 Points', color='orange')
            ax.set_xlabel('Time')
            ax.set_ylabel('Amplitude')
            ax.set_title('First 5000 Data Points')
            plt.tight_layout()
            st.pyplot(fig)
            jumlahdata = int(np.size(x))
            st.write("<h5> Jumlah data:\n </h5>",jumlahdata, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error reading file: {e}")


def preproc():
    
    import streamlit as st
    from PIL import Image
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from streamlit_option_menu import option_menu
    from scipy.signal import welch
    import plotly.graph_objects as go
    import math
    st.title("Processing Signal")
    #preproc testing
    # Define column names
    column_names = ['ECG']

    # Read the data from the text file
    data = pd.read_csv('dataecginofix1.txt', delimiter="\t", names=column_names)
    #data = pd.read_csv('dataecgvannofix.txt', delimiter="\t", names=column_names)

    # Convert the ECG column to numeric (if there are any non-numeric values, they will be coerced to NaN)
    data['ECG'] = pd.to_numeric(data['ECG'], errors='coerce')

    # Drop any rows with NaN values that may have been introduced due to conversion issues
    data.dropna(inplace=True)

    # Create sample interval and elapsed time columns
    N = len(data)
    # Define x and y
    x = np.zeros(N)
    y = np.zeros(N)
    x = np.arange(len(data)) * (1/125)
    y = data["ECG"] - data["ECG"].mean()

    N = 24400
    #N = 18900
    #N= 5000
    ecg = np.zeros(N)
    ecg_x = np.zeros(N)

    for i in range(N):
        ecg[i]=y[i]
        ecg_x[i]=x[i]

    fs = 125

    # Calculate the elapsed time
    sample_interval = np.arange(0, N)
    elapsed_time = sample_interval * (1/fs)

    #code-pythoon (hal 26)
    def dirac(x):
        if (x==0):
            dirac_delta = 1
        else:
            dirac_delta = 0
        result = dirac_delta
        return result
    h = []
    g = []
    n_list = []
    for n in range(-2,2):
        n_list.append(n)
        temp_h = 1/8 * (dirac (n-1) + 3*dirac (n) + 3*dirac (n+1)+ dirac (n+2))
        h.append(temp_h)
        temp_g= -2 * (dirac (n) - dirac (n+1))
        g.append(temp_g)




    # Hw = []
    # Gw = []
    Hw= np.zeros(50000)
    Gw = np.zeros(50000)

    i_list = []
    for i in range(0, fs+1):
        i_list.append(i)
        reG = 0
        imG = 0
        reH = 0
        imH = 0
        for k in range(-2,2):
            reG = reG + g[k+abs(-2)]*np.cos(k*2*np.pi*i/fs)
            img = imG-g[k+abs(-2)]*np.sin(k*2*np.pi*i/fs)
            reH = reH + h[k+abs(-2)]*np.cos(k*2*np.pi*i/fs)
            imH = imH - h[k+abs(-2)]*np.sin(k*2*np.pi*i/fs)
        temp_Hw = np.sqrt( (reH**2) + (imH**2) )
        temp_Gw = np.sqrt( (reG**2) + (imG**2) )
        # Hw.append(temp_Hw)
        #Gw.append(temp_Gw)
        Hw[i] = temp_Hw
        Gw[i] = temp_Gw

    i_list = i_list[0:round (fs/2)+1]

    # Create a figure with a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 5))

    # Plot h(n) as a bar chart in the first subplot
    axs[0, 0].bar(n_list, h, width=0.1)
    axs[0, 0].set_title('h(n)')
    axs[0, 0].set_xlabel('n')
    axs[0, 0].set_ylabel('h')

    # Plot g(n) as a bar chart in the second subplot
    axs[0, 1].bar(n_list, g, width=0.1, color='orange')
    axs[0, 1].set_title('g(n)')
    axs[0, 1].set_xlabel('n')
    axs[0, 1].set_ylabel('g')

    # Plot Hw as a line plot in the third subplot
    axs[1, 0].plot(i_list, Hw[0:len(i_list)])
    axs[1, 0].set_title('Hw')
    axs[1, 0].set_xlabel('Index')
    axs[1, 0].set_ylabel('Hw')

    # Plot Gw as a line plot in the fourth subplot
    axs[1, 1].plot(i_list, Gw[0:len(i_list)], color='orange')
    axs[1, 1].set_title('Gw')
    axs[1, 1].set_xlabel('Index')
    axs[1, 1].set_ylabel('Gw')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Display the plot
    st.subheader(" Visualiasi Coef Filter HPF dan LPF ")
    st.pyplot(fig)

    #hal 33

    # Create 2D Array
    Q = np.zeros((9, round (fs/2)+1))

    #ORDE 1
    i_list = []
    for i in range(0, round (fs/2)+1):
        i_list.append(i)
        Q[1][i] = Gw[i]
        Q[2][i] = Gw[2*i]*Hw[i]
        Q[3][i] = Gw[4*i]*Hw[2*i]*Hw[i]
        Q[4][i] = Gw[8*i]*Hw[4*i]*Hw[2*i]*Hw[i]
        Q[5][i] = Gw[16*i]*Hw[8*i]*Hw[4*i]*Hw[2*i]*Hw[i]
        Q[6][i] = Gw[32*i]*Hw[16*i]*Hw[8*i]*Hw[4*i]*Hw[2*i]*Hw[i]
        Q[7][i] = Gw[64*i]*Hw[32*i]*Hw[16*i]*Hw[8*i]*Hw[4*i]*Hw[2*i]*Hw[i]
        Q[8][i] = Gw[128*i]*Hw[64*i]*Hw[32*i]*Hw[16*i]*Hw[8*i]*Hw[4*i]*Hw[2*i]*Hw[i]

    # qj = np.zeros((6, 10000)) k_list = [] j = 1 a = (round (2*j) + round(2(j-1)) - 2) print('a =', a) b = (1 round(2*(j-1))) + 1 print('b', b) for k in range (a,b): k_list.append(k) qj[1][k+abs(a)] = -2 * ( dirac (k) - dirac (k+1)) plt.bar(k_list, qj[1][0:len(k_list)]) plt.show()

    #ORDE 1
    qj = np.zeros((6, 10000))

    k_list = []
    j = 1
    a = -(round (2**j) + round(2**(j-1)) - 2)
    st.subheader("Orde 1")
    st.write('a =', a)
    b = (1 - round(2**(j-1))) + 1
    st.write('b', b)
    for k in range (a,b):
        k_list.append(k)
        qj[1][k+abs(a)] = -2 * ( dirac (k) - dirac (k+1))

    #ORDE 2
    k_list = []
    j= 2
    a = -(round (2**j) + round(2**(j-1)) - 2)
    st.subheader("Orde 2")
    st.write('a', a)
    b = -(1 - round(2**(j-1))) + 1
    st.write('b', b)
    for k in range (a,b):
        k_list.append(k)
        qj[2] [k+abs(a)] = -1/4* (dirac (k-1) + 3*dirac (k) + 2*dirac (k+1) - 2*dirac (k+2)
        - 3*dirac (k+3) - dirac (k+4))

    #ORDE 3
    k_list = []
    j=3
    a=-(round(2**j) + round(2**(j-1))-2)
    st.subheader("Orde 3")
    st.write("a =", a)
    b = - (1 - round(2**(j-1))) + 1
    st.write("b =", b)
    for k in range (a,b):
        k_list.append(k)
        qj[3][k+abs(a)] = -1/32*(dirac(k-3) + 3*dirac(k-2) + 6*dirac(k-1) + 10*dirac(k)
        + 11*dirac(k+1) + 9*dirac(k+2) + 4*dirac(k+3) - 4*dirac(k+4) - 9*dirac(k+5)
        - 11*dirac(k+6) - 10*dirac(k+7) - 6*dirac(k+8) - 3*dirac(k+9) - dirac(k+10))

    k_list = []
    j=4
    a=-(round(2**j) + round(2**(j-1))-2)
    st.subheader("Orde 4")
    st.write("a =", a)
    b = - (1 - round(2**(j-1))) + 1
    st.write("b =", b)

    for k in range (a,b):
        k_list.append(k)
        qj [4][k+abs(a)] = -1/256*(dirac(k-7) + 3*dirac(k-6) + 6*dirac(k-5) + 10*dirac(k-4) + 15*dirac (k-3)
        + 21*dirac(k-2) + 28*dirac(k-1) + 36*dirac(k) + 41*dirac(k+1) + 43*dirac(k+2)
        + 42*dirac(k+3) + 38*dirac(k+4) + 31*dirac(k+5) + 21*dirac(k+6) + 8*dirac(k+7)
        - 8*dirac(k+8) - 21*dirac(k+9) - 31*dirac(k+10) - 38*dirac(k+11) - 42*dirac(k+12)
        - 43*dirac(k+13) - 41*dirac(k+14) - 36*dirac(k+15) - 28*dirac(k+16) - 21*dirac(k+17)
        - 15*dirac(k+18) - 10*dirac(k+19) - 6*dirac(k+20) - 3*dirac(k+21) - dirac(k+22))

    k_list = []
    j=5
    a=-(round(2**j) + round(2**(j-1))-2)
    st.subheader("Orde 5")
    st.write("a =", a)
    b = - (1 - round(2**(j-1))) + 1
    st.write ("b =", b)
    for k in range (a,b):
        k_list.append(k)
        qj[5][k+abs(a)] = -1/(512)*(dirac(k-15) + 3*dirac(k-14) + 6*dirac(k-13) + 10*dirac(k-12) + 15*dirac(k-11) + 21*dirac(k-10)
        + 28*dirac(k-9) + 36*dirac(k-8) + 45*dirac(k-7) + 55*dirac(k-6) + 66*dirac(k-5) + 78*dirac(k-4)
        + 91*dirac(k-3) + 105*dirac(k-2) + 120*dirac(k-1) + 136*dirac(k) + 149*dirac(k+1) + 159*dirac(k+2)
        + 166*dirac(k+3) + 170*dirac(k+4) + 171*dirac(k+5) + 169*dirac(k+6) + 164*dirac(k+7) + 156*dirac(k+8)
        + 145*dirac(k+9) + 131*dirac(k+10) + 114*dirac(k+11) + 94*dirac(k+12) + 71*dirac(k+13) + 45 *dirac(k+14)
        + 16*dirac(k+15) - 16*dirac(k+16) - 45*dirac(k+17) - 71*dirac(k+18) - 94*dirac(k+19) - 114*dirac (k+20)
        - 131*dirac(k+21) - 145*dirac(k+22) - 156*dirac(k+23) - 164*dirac(k+24) - 169*dirac(k+25)
        - 171*dirac(k+26) - 170*dirac(k+27) - 166*dirac(k+28) - 159*dirac(k+29) - 149*dirac(k+30)
        - 136*dirac(k+31) - 120*dirac(k+32) - 105*dirac(k+33) - 91*dirac(k+34) - 78*dirac(k+35)
        - 66*dirac(k+36) - 55*dirac(k+37) - 45*dirac(k+38) - 36*dirac(k+39) - 28*dirac(k+40)
        - 21*dirac(k+41) - 15*dirac(k+42) - 10*dirac(k+43) - 6*dirac(k+44) - 3*dirac(k+45)
        - dirac(k+46)) 

    #----------Q visualization
    fig, axs = plt.subplots(2, 2, figsize=(20, 10))

    # Plot h(n) as a bar chart in the first subplot
    axs[0, 0].bar(k_list, qj[1][0:len(k_list)], width=0.5)
    axs[0, 0].set_title('Orde 1')

    # Plot g(n) as a bar chart in the second subplot
    axs[0, 1].bar(k_list, qj[2][0:len(k_list)], width=0.5, color='orange')
    axs[0, 1].set_title('Orde 2')


    # Plot Hw as a line plot in the third subplot
    axs[1, 0].bar(k_list, qj[3][0:len(k_list)],width=0.5, color='red')
    axs[1, 0].set_title('Orde 3')


    # Plot Gw as a line plot in the fourth subplot
    axs[1, 1].bar(k_list, qj[4][0:len(k_list)], width=0.5, color='green')
    axs[1, 1].set_title('Orde 4')   
    st.subheader("DWT each Orde Visualize")
    st.pyplot(fig)     

    fig, ax = plt.subplots()
    for i in range(1, 9):
        line_label = "Q{}".format(i)
        ax.plot(i_list, Q[i], label=line_label)
    ax.legend()
    st.pyplot(fig)

    T1 = round (2**(1-1)) - 1
    T2 = round (2**(2-1)) - 1
    T3 = round (2**(3-1)) - 1
    T4 = round (2**(4-1)) - 1
    T5 = round (2**(5-1)) - 1

    st.write('T1 =', T1)
    st.write('T2 =', T2)
    st.write('T3 =', T3)
    st.write('T4 =', T4)
    st.write('T5 =', T5)

    #Welch Method
    min_n = 0 * fs
    max_n = 8 * fs


    def process_ecg(min_n, max_n, ecg, g, h):
        w2fm = np.zeros((5, max_n - min_n + 1))
        s2fm = np.zeros((5, max_n - min_n + 1))

        for n in range(min_n, max_n + 1):
            for j in range(1, 6):
                w2fm[j-1, n - min_n] = 0
                s2fm[j-1, n - min_n] = 0
                for k in range(-1, 3):
                    index = round(n - 2**(j-1) * k)
                    if 0 <= index < len(ecg):  # Ensure the index is within bounds
                        w2fm[j-1, n - min_n] += g[k+1] * ecg[index]  # g[k+1] to match Pascal's array index starting from -1
                        s2fm[j-1, n - min_n] += h[k+1] * ecg[index]  # h[k+1] to match Pascal's array index starting from -1

        return w2fm, s2fm

    # Compute w2fm and s2fm
    w2fm, s2fm = process_ecg(min_n, max_n, ecg, g, h)

    # Prepare data for plotting
    n_values = np.arange(min_n, max_n + 1)
    w2fm_values = [w2fm[i, :] for i in range(5)]  # Equivalent to w2fm[1,n] to w2fm[5,n] in original code (0-based index)
    s2fm_values = [s2fm[i, :] for i in range(5)]  # Equivalent to s2fm[1,n] to s2fm[5,n] in original code (0-based index)

    st.subheader("Welch processing result here.")

    #Welch each Orde
    mins = 0
    maks = 4

    w2fb = np.zeros((6,30000))
    n_list = []
    #w2fb[i] --> i = orde n_list = []
    for n in list(range(N)):
        n_list.append(n)
        for j in range(1,6):
            w2fb[1][n+T1] = 0
            w2fb[2][n+T2] = 0
            w2fb[3][n+T3] = 0
            w2fb[4][n+T4] = 0
            w2fb[5][n+T5] = 0
            a = -(round (2**j) + round(2**(j-1)) - 2 )
            b = -(1 - round(2**(j-1)))
            for k in range(a, b+1):
                if 0 <= n - (k + abs(a)) < len(ecg):
                    w2fb[1][n+T1] = w2fb[1][n+T1]+qj[1][(k+abs(a))]*ecg[n-(k+abs(a))];
                    w2fb[2][n+T2] = w2fb[2][n+T2]+qj[2][(k+abs(a))]*ecg[n-(k+abs(a))];
                    w2fb[3][n+T3] = w2fb[3][n+T3]+qj[3][(k+abs(a))]*ecg[n-(k+abs(a))];
                    w2fb[4][n+T4] = w2fb[4][n+T4]+qj[4][(k+abs(a))]*ecg[n-(k+abs(a))];
                    w2fb[5][n+T5] = w2fb[5][n+T5]+qj[5][(k+abs(a))]*ecg[n-(k+abs(a))];
                    
    figt, axs = plt.subplots(5, 1, figsize=(20, 10))
    # Plot each order in a separate subplot
    for i in range(0, 5):
        axs[i].plot(n_list, w2fb[i+1][0:len(n_list)], label=f'w2fb[{i+1}]')
        axs[i].set_title(f'Orde {i+1}')
        axs[i].set_xlabel('n')
        axs[i].set_ylabel('Amplitude')
        axs[i].legend()
    plt.tight_layout()
    st.pyplot(figt)

    #Normalize
    def norm(signal):
        max_abs = np.max(np.abs(signal))  # Find the maximum absolute value in the signal
        
        if max_abs == 0:
            return signal  # Avoid division by zero
        
        normalized_signal = signal / max_abs  # Normalize the signal
        
        return normalized_signal
    #Gradien Calculate
    gradien1 = np.zeros(len(ecg))
    gradien2 = np.zeros(len(ecg))
    gradien3 = np.zeros(len(ecg))

    # Define delay
    delay = T3

    # Compute gradien3
    N = len(ecg)
    for k in range(delay, N - delay):
        gradien3[k] = w2fb[3][k - delay] - w2fb[3][k + delay]

    gradien3 = gradien3

    # Plot the data using Matplotlib
    figu, axs=plt.subplots(figsize=(15, 4))
    axs.plot(gradien3[0:N], color='blue', label='Gradien 3')
    # Add titles and labels
    axs.set_title('Gradien 3')
    axs.set_xlabel('Time (s)')
    axs.set_ylabel('Amplitude (V)')
    plt.legend()
    st.pyplot(figu)

    #QRS Detection
    # Function to normalize the signal
    def normalize_signal(signal):
        max_abs = np.max(np.abs(signal))  # Find the maximum absolute value in the signal
        if max_abs == 0:
            return signal  # Avoid division by zero
        normalized_signal = signal / max_abs  # Normalize the signal
        return normalized_signal

    # Initialize pulse_QRS and gradien
    pulse_QRS = np.zeros(N)
    gradien = np.zeros(N)

    # Compute gradien
    for k in range(T3, N - T3):
        gradien[k] = w2fb[3][k - T3] - w2fb[3][k + T3]

    # Normalize gradien
    gradien = normalize_signal(gradien)

    # QRS detection logic
    for i in range(N):
        if gradien[i] > 0.5:
            pulse_QRS[i - T3] = 1
        else:
            pulse_QRS[i - T3] = 0

    # Create the figure
    fig = go.Figure()

    # Add ECG signal trace
    fig.add_trace(go.Scatter(x=elapsed_time[0:5000], y=gradien[0:N], mode='lines', name='ECG', line=dict(color='red')))

    # Add QRS detection trace
    fig.add_trace(go.Scatter(x=elapsed_time[0:5000], y=pulse_QRS[0:N], mode='lines', name='QRS Detection', line=dict(color='blue')))

    # Update layout
    fig.update_layout(
        title='Gradient at DWT Order 3',
        xaxis_title='Time (s)',
        yaxis_title='Amplitude (V)',
        height=400,
        width=1500,
    )
    fig.update_layout(legend=dict(x=1, y=1, traceorder='normal', font=dict(size=12)))

    # Show the figure in Streamlit
    st.plotly_chart(fig)


    #calculation
    ptp=0
    waktu=np.zeros(np.size(pulse_QRS))
    selisih=np.zeros(np.size(pulse_QRS))
    for n in range (np.size(pulse_QRS)-1):
        if pulse_QRS[n]<pulse_QRS[n+1]:
            waktu[ptp]=n/fs;
            selisih[ptp]=waktu[ptp]-waktu[ptp-1];
            ptp+=1
    ptp=ptp-1

    #Nilai J
    j=0
    peak=np.zeros(np.size(pulse_QRS))
    for n in range(np.size(pulse_QRS)-1):
        if (pulse_QRS[n]==1) and (pulse_QRS[n-1]==0):
            peak[j] =n
            j+=1

    #nilai rata
    temp=0
    interval=np.zeros(np.size(pulse_QRS))
    BPM=np.zeros(np.size(pulse_QRS))
    for n in range (1, ptp):
        interval[n]=(peak[n]-peak[n-1])*(1/fs)
        BPM[n]=60/interval[n]
        temp=temp+BPM[n]
        rata=temp/(n)

    #nilai RR_SDNN
    RR_SDNN=0
    for n in range(ptp):
        RR_SDNN += (((selisih[n]-(60/rata)))**2)

    SDNN = math.sqrt(RR_SDNN/(ptp-1))

    #nilai RMSDD
    RR_RMSDD =0
    for n in range (ptp):
        RR_RMSDD +=((selisih[n+1]-selisih[n])**2)

    RMSDD =math.sqrt(RR_RMSDD/(ptp-1))

    #nilai pNN50
    NN50=0

    for n in range (ptp):
        if(abs(selisih[n+1]-selisih[n])>0.05):
            NN50 +=1

    pNN50 = (NN50/(ptp-1))*100

    #nilai SDSD
    dif =0
    for n in range(ptp):
        dif += abs(selisih[n]-selisih[n+1])
    RRdif = dif/(ptp-1)

    RR_SDSD =0
    for n in range (ptp):
        RR_SDSD += ((abs(selisih[n]-selisih[n+1])-RRdif)**2)

    SDSD = math.sqrt(RR_SDSD/(ptp-2))

    # Create DataFrame
    data = {
        "Metrics": ["PTP", "Peak", "Rata-rata", "SDNN", "RMSSD", "pNN50", "SDSD"],
        "Values": [ptp, j, rata, SDNN, RMSDD, pNN50, SDSD]
    }

    df = pd.DataFrame(data)

    # Display table in Streamlit
    st.table(df)

    # Tachogram visualization
    import plotly.express as px

    # Assume pulse_QRS and selisih are defined previously
    # Tachogram visualization
    bpm_rr = np.zeros(ptp)
    for n in range(ptp):
        bpm_rr[n] = 60 / selisih[n]
        if bpm_rr[n] > 100:
            bpm_rr[n] = 100  # 100 maksimal bpm normal


    # Plot Tachogram
    n = np.arange(0, ptp, 1, dtype=int)
    fig_tachogram = go.Figure()
    fig_tachogram.add_trace(go.Scatter(x=n, y=bpm_rr, mode='lines', name='Tachogram', line=dict(color='red')))
    fig_tachogram.update_layout(
        title={
            'text': "Tachogram",
            'font': {'color': 'black'}
        },
        xaxis_title={
            'text': "Indeks",
            'font': {'color': 'black'}
        },
        yaxis_title={
            'text': "BPM (per minute)",
            'font': {'color': 'black'}
        },
        height=400,
        width=1500,
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            tickfont=dict(color='black')  # Menambah properti tickfont untuk sumbu-x
        ),
        yaxis=dict(
            tickfont=dict(color='black')  # Menambah properti tickfont untuk sumbu-y
        )
    )
    st.plotly_chart(fig_tachogram)

    #plot poincare
    import streamlit as st
    import numpy as np
    import matplotlib.pyplot as plt

   
    # INTERVAL RR
    markR = np.zeros(N)
    for i in range(N-1):
        if (pulse_QRS[i]==0) & (pulse_QRS[i+1]==1):
            markR[i] = 1

    RRpos = []
    for i in range(N):
        if markR[i] == 1:
            RRpos.append(i)

    RRintervals = []
    for i in range(len(RRpos)-1):
        RRintervals.append(RRpos[i+1]-RRpos[i])

    RRdistance = []
    for i in range(len(RRpos)-1):
        RRdistance.append((RRpos[i+1]-RRpos[i])/fs)

    # Streamlit app
    st.title("Interval RR Analysis")

    st.subheader("RR Intervals")
    st.write(RRintervals)

    # Plot RR intervals
    fig, ax = plt.subplots()
    ax.plot(RRdistance)
    ax.set_title("Interval RR")
    ax.set_xlabel("Index of PtP")
    ax.set_ylabel("Time (s)")
    st.pyplot(fig)

    # Scatter plot of RR intervals
    fig, ax = plt.subplots(figsize=(5, 5))
    for i in range(len(RRintervals)-1):
        ax.scatter(RRintervals[i], RRintervals[i+1], color='b')
    ax.set_xlabel('RR interval at index i')
    ax.set_ylabel('RR interval at index i+1')
    ax.set_title('Scatter plot of RR intervals')
    ax.grid(True)
    st.pyplot(fig)

    # Downshift baseline
    bpm_rr_downshift = bpm_rr - 65

    # Histogram
    fig_histogram = px.histogram(bpm_rr_downshift, nbins=100, title="Histogram", labels={'value': 'BPM (per minute)'})
    fig_histogram.update_layout(
        title={
            'text': "Histogram",
            'font': {'color': 'black'}
        },
        xaxis_title={
            'text': "BPM (per minute)",
            'font': {'color': 'black'}
        },
        yaxis_title={
            'text': "Jumlah data",
            'font': {'color': 'black'}
        },
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            tickfont=dict(color='black')  # Menambah properti tickfont untuk sumbu-x
        ),
        yaxis=dict(
            tickfont=dict(color='black')  # Menambah properti tickfont untuk sumbu-y
        )
    )
    st.plotly_chart(fig_histogram)

    # Downshift Baseline BPM
    fig_downshift = go.Figure()
    fig_downshift.add_trace(go.Scatter(x=n, y=bpm_rr_downshift, mode='lines', name='Downshift Baseline BPM', line=dict(color='blue')))
    fig_downshift.update_layout(
        title={
            'text': "Downshift Baseline BPM",
            'font': {'color': 'black'}
        },
        xaxis_title={
            'text': "Index",
            'font': {'color': 'black'}
        },
        yaxis_title={
            'text': "Amplitude of BPM",
            'font': {'color': 'black'}
        },
        height=400,
        width=1500,
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            tickfont=dict(color='black')  # Menambah properti tickfont untuk sumbu-x
        ),
        yaxis=dict(
            tickfont=dict(color='black')  # Menambah properti tickfont untuk sumbu-y
        )
    )
    st.plotly_chart(fig_downshift)

    #Segmentation
    n = 7  # Jumlah segmentasi

    def divide_signal(signal, n):
        signal_length = len(signal)
        segment_length = signal_length // n
        overlap_length = int(segment_length * 0.2)  # Lebar overlap
        
        segments = []
        start = 0

        for i in range(n):
            if i != n - 1:
                end = start + segment_length + overlap_length
                segments.append(signal[start:end])
                start += segment_length - overlap_length  
            else:
                segments.append(signal[start:])
                
        indices = np.arange(segment_length)  # Indices for x-axis 

        return segments, indices

    def apply_hamming_window(segment):
        M = len(segment) - 1
        hamming_window = np.zeros(M+1)
        for i in range(M+1):
            hamming_window[i] = 0.54 - 0.46 * np.cos(2 * np.pi * i / M)
        return segment * hamming_window

    segments, indices = divide_signal(bpm_rr, n)
    wndw_segments = [apply_hamming_window(segment) for segment in segments]

    # Menggunakan Streamlit untuk plot
    st.subheader('Segmentasi Sinyal dengan Jendela Hamming')

    fig, axes = plt.subplots(n, 1, figsize=(12, n * 2))
    for i in range(n):
        axes[i].plot(segments[i], label=f'Segment {i + 1}')
        axes[i].plot(wndw_segments[i], color='red')
        axes[i].set_title(f'Segment {i + 1}')
        axes[i].set_xlabel('Indeks')
        axes[i].set_ylabel('Amplitudo')
        axes[i].legend()
        axes[i].grid(True)

    plt.tight_layout()
    st.pyplot(fig)


    import streamlit as st
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import welch

    # Function to calculate Fourier Transform
    def fourier_transform(signal):
        N = len(signal)
        fft_result = np.zeros(N, dtype=complex)
        for k in range(N):
            for n in range(N):
                fft_result[k] += signal[n] * np.exp(-2j * np.pi * k * n / N)
        return fft_result

    # Function to calculate frequency axis
    def calculate_frequency(N, sampling_rate):
        return np.arange(N) * sampling_rate / N

    # Function to divide signal into segments
    def divide_signal(signal, n):
        signal_length = len(signal)
        segment_length = signal_length // n
        overlap_length = int(segment_length * 0.2)  # Overlap length
        
        segments = []
        start = 0

        for i in range(n):
            if i != n - 1:
                end = start + segment_length + overlap_length
                segments.append(signal[start:end])
                start += segment_length - overlap_length  
            else:
                segments.append(signal[start:])
                
        indices = np.arange(segment_length)  # Indices for x-axis 

        return segments, indices

    # Function to apply Hamming window
    def apply_hamming_window(segment):
        M = len(segment) - 1
        hamming_window = np.zeros(M+1)
        for i in range(M+1):
            hamming_window[i] = 0.54 - 0.46 * np.cos(2 * np.pi * i / M)
        return segment * hamming_window

    # Function for manual interpolation
    def manual_interpolation(x, xp, fp):
        return np.interp(x, xp, fp)


    segments, indices = divide_signal(bpm_rr, n)
    wndw_segments = [apply_hamming_window(segment) for segment in segments]

    # Ensure all segments have the same length
    min_len = min(len(segment) for segment in wndw_segments)
    wndw_segments = [segment[:min_len] for segment in wndw_segments]

    # Compute accumulated FFT
    fft_total = np.zeros(min_len // 2)

    for segment in wndw_segments:
        fft_result = fourier_transform(segment)
        fft_freq = calculate_frequency(len(segment), 1)  # 1 Hz sampling rate for simplicity
        half_point = len(fft_freq) // 2
        fft_result_half = np.abs(fft_result[:half_point])
        fft_total += fft_result_half

    # # Compute Welch's periodogram
    # f, Pxx = welch(np.concatenate(wndw_segments), fs=1, nperseg=1024)  # fs=1 for simplicity

    # Plot the accumulated FFT
    st.subheader('Accumulated FFT')
    fig, ax = plt.subplots()
    ax.plot(fft_freq[:min_len // 2], fft_total, linestyle='-', color='r')
    ax.set_title('Accumulated FFT')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude')
    ax.grid(True)
    st.pyplot(fig)

    # Frequency bands
    x_vlf = np.linspace(0.003, 0.04, 99)
    x_lf = np.linspace(0.04, 0.15, 99)
    x_hf = np.linspace(0.15, 0.4, 99)

    # Compute the interpolated values manually
    y_vlf = manual_interpolation(x_vlf, fft_freq[:min_len // 2], np.abs(fft_total))
    y_lf = manual_interpolation(x_lf, fft_freq[:min_len // 2], np.abs(fft_total))
    y_hf = manual_interpolation(x_hf, fft_freq[:min_len // 2], np.abs(fft_total))

    # Plotting the FFT spectrum with the interpolated values
    st.subheader("FFT Spectrum (Welch's Periodogram)")
    fig, ax = plt.subplots(figsize=(20, 4))
    ax.plot(fft_freq[:min_len // 2], np.abs(fft_total), color="k", linewidth=0.3)
    ax.set_title("FFT Spectrum (Welch's Periodogram)")

    # Fill between the frequency bands
    ax.fill_between(x_vlf, y_vlf, alpha=0.2, color="#A651D8", label="VLF")
    ax.fill_between(x_lf, y_lf, alpha=0.2, color="#51A6D8", label="LF")
    ax.fill_between(x_hf, y_hf, alpha=0.2, color="#D8A651", label="HF")

    ax.set_xlim(0, 0.5)
    ax.set_ylim(0)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Density")
    ax.legend()
    st.pyplot(fig)


    # Calculation functions
    def calculate_mpf(frequencies, power_spectrum):
        mpf = np.sum(frequencies * power_spectrum) / np.sum(power_spectrum)
        return mpf

    def estimate_respiratory_rate(mpf):
        respiratory_rate = mpf * 60
        return respiratory_rate


    # Inputs
    st.title("Respiratory Rate and Power Distribution Calculator")

    # Compute PSD
    frequencies, power_spectrum = x_hf, y_hf

    # Calculate MPF and Respiratory Rate
    mpf = calculate_mpf(frequencies, power_spectrum)
    respiratory_rate = estimate_respiratory_rate(mpf)

    st.write(f"Estimated Respiratory Rate: {respiratory_rate:.2f} breaths per minute")
    def trapezoidal_rule_integral(y, x):
        area = 0.0
        n = len(x)
        for i in range(1, n):
            width = x[i] - x[i - 1]
            height = (y[i] + y[i - 1]) / 2.0
            area += width * height
        return area


    # Compute integrals
    fft_freq_half = fft_freq[:half_point]
    time = fft_freq_half
    data = np.abs(fft_total)
    totalPower = trapezoidal_rule_integral(data, time)


    vlf = trapezoidal_rule_integral(y_vlf, x_vlf) / totalPower
    lf = trapezoidal_rule_integral(y_lf, x_lf) / totalPower
    hf = trapezoidal_rule_integral(y_hf, x_hf) / totalPower

    st.write(f"VLF: {vlf:.2f}")
    st.write(f"LF: {lf:.2f}")
    st.write(f"HF: {hf:.2f}")
    totalPower = 1
    st.write(f"Total power: {totalPower:.2f}")

    # Plot the bar chart
    labels = ['VLF', 'LF', 'HF', 'Total Power']
    values = [vlf, lf, hf, totalPower]

    fig, ax = plt.subplots()
    ax.bar(labels, values, color=['blue', 'green', 'red', 'purple'])
    ax.set_xlabel('Frequency Bands')
    ax.set_ylabel('Power')
    ax.set_title('Power Distribution in Different Frequency Bands')
    st.pyplot(fig)

    # Ratio and normalized values
    ratio = lf / hf
    nuLF = lf / (lf + hf) * 100
    nuHF = hf / (lf + hf) * 100

    st.write(f"Ratio LF/HF: {ratio:.2f}")
    st.write(f"Normalized LF: {nuLF:.2f}")
    st.write(f"Normalized HF: {nuHF:.2f}")

    # Scatter plot
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(nuLF, nuHF, color='blue', s=100, label='LF/HF Ratio Point')
    ax.set_xlabel('LF (nu)')
    ax.set_ylabel('HF (nu)')
    ax.set_title('Autonomic Balance Diagram')

    ax.axhline(33.3, color='gray', linestyle='--')
    ax.axhline(66.6, color='gray', linestyle='--')
    ax.axvline(33.3, color='gray', linestyle='--')
    ax.axvline(66.6, color='gray', linestyle='--')

    ax.text(nuLF, nuHF, f' LF/HF: {ratio:.2f}\n LF (nu): {nuLF:.2f}\n HF (nu): {nuHF:.2f}', fontsize=10, ha='right')

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    areas = ['7', '8', '9',
            '4', '5', '6',
            '1', '2', '3']

    positions = [(16.65, 83.3), (50, 83.3), (83.3, 83.3),
                (16.65, 50), (50, 50), (83.3, 50),
                (16.65, 16.65), (50, 16.65), (83.3, 16.65)]

    for pos, label in zip(positions, areas):
        ax.text(pos[0], pos[1], label, fontsize=8, ha='center', va='center')

    ax.legend()
    st.pyplot(fig)

def dashboard_page():
    
    #Sidebar_Option_Menu
    with st.sidebar:
        selected = option_menu(
            menu_title="ASN Group 2",
            options = ["Home","Pre-processing","Analysis","Calculation"],
            menu_icon=None,
            icons=["house","hr","reception-4","calculator"],
            default_index=0,
            # orientation="horizontal",
            styles={
                "icon": {"color":"orange","font-size": "25px"},
                "nav-link":{"text-align:" : "left","margin":"10px","--hover-color":"#eee","font-size": "20px"
                }
            }
        )
    if selected == "Home":
        plot_data()
    if selected == "Pre-processing":
        preproc()
    if selected == "Analysis":
        analysis()
    if selected == "Calculation":
        calculation()
    
    
    
# Main App
def main():
    if "page" not in st.session_state:
        st.session_state.page = "Landing"
    
    if st.session_state.page == "Dashboard":
        dashboard_page()
    else:
        landing_page()

if __name__ == "__main__":
    main()
