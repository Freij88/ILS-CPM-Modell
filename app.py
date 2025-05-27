import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import io
import json

# Konfiguration av sidan
st.set_page_config(
    page_title="CPM-modell för ILS-mjukvaror - Dynamisk ROC-analys",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialisera session state variabler"""
    if 'csf_list' not in st.session_state:
        st.session_state.csf_list = [
            "Efterlevnad av ILS-ramverk",
            "Pris för kund",
            "Tidsbesparing",
            "Skalbarhet drift",
            "Informationssäkerhetsklassning",
            "Skalbarhet AI",
            "Funktionell bredd inom ILS",
            "Förmåga att tolka och hantera olika indataformat",
            "Supportkostnad",
            "Output - Struktur",
            "Grad av automation",
            "Time-to-deploy",
            "Systemintegration",
            "Robusthet",
            "Output - Filformat",
            "Användarvänlighet (UI/UX)",
            "Kundbas",
            "Utbildningsbehov",
            "Övrig funktionalitet"
        ]
    if 'vendor_list' not in st.session_state:
        st.session_state.vendor_list = ["Combitech", "Konkurrent A", "Konkurrent B"]
    if 'ratings_df' not in st.session_state:
        st.session_state.ratings_df = pd.DataFrame()
    if 'csf_order' not in st.session_state:
        st.session_state.csf_order = list(range(len(st.session_state.csf_list)))

def calculate_roc_weights(n_csfs, order):
    """
    Beräkna ROC-vikter (Rank Order Centroid) för givna CSF:er
    
    Args:
        n_csfs: Antal CSF:er
        order: Lista med prioritetsordning (0-indexerad position för varje CSF)
    
    Returns:
        List med normaliserade vikter
    """
    if n_csfs == 0:
        return []
    
    # Debug information
    debug_info = {
        'n_csfs': n_csfs,
        'input_order': order.copy(),
        'calculations': []
    }
    
    # Skapa rankning: position i order-listan bestämmer rankingen
    # order[i] anger vilken position CSF i har i prioritetsordningen
    # Vi behöver invertera detta för att få rankingen
    ranks = [0] * n_csfs
    
    # Sortera för att få rätt rankning
    sorted_positions = sorted(enumerate(order), key=lambda x: x[1])
    
    for rank_1_indexed, (csf_index, _) in enumerate(sorted_positions, 1):
        ranks[csf_index] = rank_1_indexed
    
    debug_info['ranks'] = ranks.copy()
    
    # Beräkna ROC-vikter enligt formeln: w_i = (1/N) * sum(1/k för k från rank_i till N)
    weights = []
    for i in range(n_csfs):
        rank = ranks[i]
        # ROC-formel: summa från rank till N
        harmonic_sum = sum(1/k for k in range(rank, n_csfs + 1))
        weight = (1/n_csfs) * harmonic_sum
        weights.append(weight)
        
        debug_info['calculations'].append({
            'csf_index': i,
            'rank': rank,
            'harmonic_sum': harmonic_sum,
            'weight_before_normalization': weight
        })
    
    # Normalisera så summan blir exakt 1.0
    total_weight = sum(weights)
    if total_weight > 0:
        normalized_weights = [w / total_weight for w in weights]
    else:
        # Fallback till lika vikter
        normalized_weights = [1/n_csfs] * n_csfs
    
    debug_info['total_weight_before_norm'] = total_weight
    debug_info['normalized_weights'] = normalized_weights.copy()
    debug_info['final_sum'] = sum(normalized_weights)
    
    # Lagra debug info i session state för visning
    st.session_state.roc_debug = debug_info
    
    return normalized_weights

def get_current_csf_data():
    """Hämta aktuell CSF-data med ROC-viktning"""
    csf_names = st.session_state.csf_list
    order = st.session_state.csf_order
    weights = calculate_roc_weights(len(csf_names), order)
    
    # Skapa rankning för visning
    ranks = [0] * len(csf_names)
    sorted_positions = sorted(enumerate(order), key=lambda x: x[1])
    
    for rank_1_indexed, (csf_index, _) in enumerate(sorted_positions, 1):
        ranks[csf_index] = rank_1_indexed
    
    return [
        {"name": csf_names[i], "weight": weights[i], "rank": ranks[i]}
        for i in range(len(csf_names))
    ]

def initialize_ratings_dataframe():
    """Initialisera eller uppdatera ratings DataFrame"""
    csf_data = get_current_csf_data()
    vendors = st.session_state.vendor_list
    
    # Skapa ny DataFrame med aktuella CSF:er och vendors
    new_df = pd.DataFrame(index=vendors, columns=[csf["name"] for csf in csf_data])
    
    # Fyll med befintliga värden om de finns
    if not st.session_state.ratings_df.empty:
        for vendor in vendors:
            for csf in csf_data:
                if (vendor in st.session_state.ratings_df.index and 
                    csf["name"] in st.session_state.ratings_df.columns):
                    new_df.loc[vendor, csf["name"]] = st.session_state.ratings_df.loc[vendor, csf["name"]]
    
    # Fyll tomma celler med 3 (standardbetyg)
    new_df = new_df.fillna(3).infer_objects(copy=False)
    st.session_state.ratings_df = new_df

def calculate_results():
    """Beräkna viktade resultat baserat på aktuella betyg och ROC-vikter"""
    if st.session_state.ratings_df.empty:
        return pd.DataFrame()
    
    csf_data = get_current_csf_data()
    vendors = st.session_state.vendor_list
    
    results = []
    for vendor in vendors:
        raw_sum = 0
        weighted_sum = 0
        
        for csf in csf_data:
            if csf["name"] in st.session_state.ratings_df.columns:
                rating = st.session_state.ratings_df.loc[vendor, csf["name"]]
                raw_sum += rating
                weighted_sum += rating * csf["weight"]
        
        # Normaliserad poäng (0-100 skala)
        normalized_score = (weighted_sum / 4.0) * 100
        
        results.append({
            "Vendor": vendor,
            "Raw Sum": round(raw_sum, 2),
            "Weighted Sum": round(weighted_sum, 3),
            "Normalized (0-100)": round(normalized_score, 1)
        })
    
    return pd.DataFrame(results)

def create_bar_chart(results_df):
    """Skapa stapeldiagram över viktade resultat"""
    if results_df.empty:
        return None
    
    fig = px.bar(
        results_df, 
        x="Vendor", 
        y="Weighted Sum",
        title="Viktade totalpoäng per leverantör",
        color="Weighted Sum",
        color_continuous_scale="Viridis",
        text="Weighted Sum"
    )
    
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(height=400, showlegend=False, yaxis=dict(range=[0, 4]))
    
    return fig

def create_heatmap():
    """Skapa heatmap över alla betyg"""
    if st.session_state.ratings_df.empty:
        return None
    
    # Konvertera DataFrame till numerisk för heatmap
    df_numeric = st.session_state.ratings_df.astype(float)
    
    fig = go.Figure(data=go.Heatmap(
        z=df_numeric.values,
        x=df_numeric.columns,
        y=df_numeric.index,
        colorscale='RdYlGn',
        zmin=1,
        zmax=4,
        text=df_numeric.values,
        texttemplate="%{text}",
        textfont={"size": 12},
        colorbar=dict(title="Poäng")
    ))
    
    fig.update_layout(
        title="Heatmap: Alla betyg per CSF och leverantör",
        xaxis_title="Kritiska Framgångsfaktorer (CSF)",
        yaxis_title="Leverantörer", 
        height=400
    )
    
    return fig

def export_to_csv(results_df, csf_data):
    """Exportera alla data till CSV"""
    output = io.StringIO()
    
    # Exportera CSF-viktningar
    output.write("=== CSF VIKTNINGAR (ROC-metod) ===\n")
    csf_df = pd.DataFrame(csf_data)
    csf_df.to_csv(output, index=False, sep=';')
    
    output.write("\n=== BETYG PER LEVERANTÖR OCH CSF ===\n")
    st.session_state.ratings_df.to_csv(output, sep=';')
    
    output.write("\n=== SAMMANFATTADE RESULTAT ===\n")
    results_df.to_csv(output, index=False, sep=';')
    
    return output.getvalue()

def show_roc_debug():
    """Visa debug-information för ROC-beräkningar"""
    if 'roc_debug' not in st.session_state:
        return
    
    debug = st.session_state.roc_debug
    
    with st.expander("🔍 ROC-beräkningsdetaljer (Debug)", expanded=False):
        st.write("**Indata:**")
        st.write(f"- Antal CSF:er: {debug['n_csfs']}")
        st.write(f"- Prioritetsordning (input): {debug['input_order']}")
        st.write(f"- Beräknade ranker: {debug['ranks']}")
        
        st.write("**Viktberäkningar:**")
        calc_df = pd.DataFrame(debug['calculations'])
        if not calc_df.empty:
            calc_df['CSF_namn'] = [st.session_state.csf_list[i] for i in calc_df['csf_index']]
            calc_df = calc_df[['CSF_namn', 'rank', 'harmonic_sum', 'weight_before_normalization']]
            st.dataframe(calc_df, use_container_width=True)
        
        st.write("**Normalisering:**")
        st.write(f"- Summa före normalisering: {debug['total_weight_before_norm']:.6f}")
        st.write(f"- Summa efter normalisering: {debug['final_sum']:.6f}")
        
        if abs(debug['final_sum'] - 1.0) > 1e-10:
            st.warning(f"⚠️ Vikternas summa är inte exakt 1.0: {debug['final_sum']:.10f}")
        else:
            st.success("✅ Vikternas summa är korrekt (1.0)")

def main():
    initialize_session_state()
    
    st.title("📊 CPM-modell för ILS-mjukvaror")
    st.markdown("### Dynamisk ROC-analys med anpassningsbara CSF:er och leverantörer")
    
    st.markdown("""
    **Denna applikation erbjuder:**
    - Dynamiska CSF:er (lägg till/ta bort kritiska framgångsfaktorer)
    - Automatisk ROC-viktberäkning (Rank Order Centroid)
    - Flexibel leverantörslista
    - Realtidsuppdatering av resultat
    - Export till CSV för vidare analys
    """)
    
    # Sidebar för konfiguration
    with st.sidebar:
        st.header("⚙️ Konfiguration")
        
        # CSF-hantering
        st.subheader("📋 Kritiska Framgångsfaktorer (CSF)")
        
        # Lägg till ny CSF
        new_csf = st.text_input("Lägg till ny CSF:")
        if st.button("➕ Lägg till CSF") and new_csf and new_csf not in st.session_state.csf_list:
            st.session_state.csf_list.append(new_csf)
            st.session_state.csf_order.append(len(st.session_state.csf_list) - 1)
            initialize_ratings_dataframe()
            st.success(f"Lade till: {new_csf}")
            st.rerun()
        
        # Ta bort CSF
        if len(st.session_state.csf_list) > 1:
            csf_to_remove = st.selectbox("Ta bort CSF:", [""] + st.session_state.csf_list)
            if st.button("🗑️ Ta bort CSF") and csf_to_remove:
                idx = st.session_state.csf_list.index(csf_to_remove)
                st.session_state.csf_list.remove(csf_to_remove)
                # Uppdatera order array korrekt
                st.session_state.csf_order.pop(idx)
                # Justera index för CSF:er som kommer efter den borttagna
                st.session_state.csf_order = [pos if pos < idx else pos - 1 for pos in st.session_state.csf_order]
                initialize_ratings_dataframe()
                st.success(f"Tog bort: {csf_to_remove}")
                st.rerun()
        
        # CSF-prioritering för ROC-vikter
        st.subheader("🎯 CSF-prioritering (för ROC-vikter)")
        st.write("Ange prioritetsordning (1 = högst prioritet):")
        
        # Visa aktuell ordning och låt användaren ändra
        csf_data = get_current_csf_data()
        
        # Sortera efter nuvarande rank för visning
        sorted_csfs = sorted(csf_data, key=lambda x: x["rank"])
        
        reorder_needed = False
        for csf in sorted_csfs:
            csf_idx = st.session_state.csf_list.index(csf["name"])
            current_rank = csf["rank"]
            
            new_rank = st.number_input(
                f"{csf['name'][:40]}..." if len(csf['name']) > 40 else csf['name'],
                min_value=1,
                max_value=len(st.session_state.csf_list),
                value=current_rank,
                key=f"rank_{csf['name']}"
            )
            
            if new_rank != current_rank:
                # Uppdatera prioritetsordning
                st.session_state.csf_order[csf_idx] = new_rank - 1
                reorder_needed = True
        
        if reorder_needed:
            st.rerun()
        
        st.divider()
        
        # Leverantörshantering
        st.subheader("🏢 Leverantörer")
        
        # Lägg till ny leverantör
        new_vendor = st.text_input("Lägg till ny leverantör:")
        if st.button("➕ Lägg till leverantör") and new_vendor and new_vendor not in st.session_state.vendor_list:
            st.session_state.vendor_list.append(new_vendor)
            initialize_ratings_dataframe()
            st.success(f"Lade till: {new_vendor}")
            st.rerun()
        
        # Ta bort leverantör
        if len(st.session_state.vendor_list) > 1:
            vendor_to_remove = st.selectbox("Ta bort leverantör:", [""] + st.session_state.vendor_list)
            if st.button("🗑️ Ta bort leverantör") and vendor_to_remove:
                st.session_state.vendor_list.remove(vendor_to_remove)
                initialize_ratings_dataframe()
                st.success(f"Tog bort: {vendor_to_remove}")
                st.rerun()
    
    # Initialisera ratings DataFrame
    initialize_ratings_dataframe()
    
    # Visa aktuella CSF-vikter
    st.header("📊 Aktuella CSF-vikter (ROC-metod)")
    csf_data = get_current_csf_data()
    csf_df = pd.DataFrame(csf_data)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(csf_df, use_container_width=True)
        
        # Visa debug-information
        show_roc_debug()
        
    with col2:
        # Viktfördelning som cirkeldiagram
        fig_pie = px.pie(
            csf_df, 
            values='weight', 
            names='name', 
            title="Viktfördelning"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Poängsättning
    st.header("🎯 Poängsättning")
    st.markdown("**Sätt betyg för varje leverantör och CSF (1 = Mycket dålig, 4 = Mycket bra)**")
    
    # Organisera poängsättning i kolumner
    cols = st.columns(len(st.session_state.vendor_list))
    
    for col_idx, vendor in enumerate(st.session_state.vendor_list):
        with cols[col_idx]:
            st.subheader(f"🏢 {vendor}")
            
            for csf in csf_data:
                current_rating = st.session_state.ratings_df.loc[vendor, csf["name"]]
                
                # Skapa kort CSF-namn för bättre layout
                short_name = csf['name'][:25] + "..." if len(csf['name']) > 25 else csf['name']
                st.write(f"**{short_name}**")
                st.caption(f"Vikt: {csf['weight']:.3f} | Rank: {csf['rank']}")
                
                # Knappar för poängsättning
                button_cols = st.columns(4)
                rating_labels = ["1️⃣ Mycket dålig", "2️⃣ Dålig", "3️⃣ Bra", "4️⃣ Mycket bra"]
                
                new_rating = current_rating
                for i, (btn_col, label) in enumerate(zip(button_cols, rating_labels)):
                    with btn_col:
                        rating_value = i + 1
                        button_type = "primary" if current_rating == rating_value else "secondary"
                        
                        if st.button(
                            f"{rating_value}",
                            key=f"{vendor}_{csf['name']}_{rating_value}",
                            type=button_type,
                            help=label
                        ):
                            new_rating = rating_value
                
                # Uppdatera rating om den ändrats
                if new_rating != current_rating:
                    st.session_state.ratings_df.loc[vendor, csf["name"]] = new_rating
                    st.rerun()
                
                st.divider()
    
    # Resultat och visualiseringar
    st.header("📈 Resultat och Analys")
    
    results_df = calculate_results()
    
    if not results_df.empty:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📊 Sammanfattning")
            st.dataframe(results_df, use_container_width=True)
            
            # Export-knapp
            csv_data = export_to_csv(results_df, csf_data)
            st.download_button(
                label="📁 Exportera till CSV",
                data=csv_data,
                file_name="cpm_analys_resultat.csv",
                mime="text/csv"
            )
        
        with col2:
            st.subheader("📈 Stapeldiagram")
            bar_chart = create_bar_chart(results_df)
            if bar_chart:
                st.plotly_chart(bar_chart, use_container_width=True)
        
        # Heatmap över alla betyg
        st.subheader("🗺️ Heatmap - Alla betyg")
        heatmap = create_heatmap()
        if heatmap:
            st.plotly_chart(heatmap, use_container_width=True)
    else:
        st.info("Inga resultat att visa. Sätt betyg för leverantörerna ovan.")

if __name__ == "__main__":
    main()
