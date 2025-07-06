import streamlit as st
import pandas as pd
import plotly.express as px
import psycopg2

st.set_page_config(layout="wide", page_title="Volatility Smile Live")
st.title("🔮 Live Volatility Smile Predictions")

@st.cache_data(ttl=5)
def load_data():
    # --- Connexion à la base ---
    conn = psycopg2.connect(
        host="timescaledb",
        port="5432",
        user="khalil",
        password="MyStrongPass123",
        dbname="volatility_db"
    )

    # --- Charger les 40 dernières prédictions ---
    query = """
        SELECT * FROM predicted_smile
        ORDER BY ts_utc DESC
        LIMIT 40;
    """
    predicted_df = pd.read_sql(query, conn)

    if predicted_df.empty:
        conn.close()
        return pd.DataFrame(), pd.DataFrame()

    # --- Supprimer les doublons par (maturity, strike, right)
    predicted_df = predicted_df.drop_duplicates(subset=["maturity", "strike", "right"])

    # --- Obtenir IV réel correspondant (même maturity, timestamp max)
    maturity = predicted_df["maturity"].iloc[0]
    actual_query = f"""
        SELECT 
            time AT TIME ZONE 'UTC' AS ts_utc,
            maturity,
            strike,
            call_iv,
            put_iv
        FROM spy_option_chain
        WHERE maturity = '{maturity}'
        AND (call_iv IS NOT NULL OR put_iv IS NOT NULL)
        AND time = (
            SELECT MAX(time) FROM spy_option_chain WHERE maturity = '{maturity}'
        )
        ORDER BY strike;
    """
    actual_df_raw = pd.read_sql(actual_query, conn)
    conn.close()

    # --- Formater actual IVs ---
    if not actual_df_raw.empty:
        calls = actual_df_raw[['ts_utc', 'maturity', 'strike', 'call_iv']].copy()
        calls['right'] = 'C'
        calls = calls.rename(columns={'call_iv': 'actual_iv'})

        puts = actual_df_raw[['ts_utc', 'maturity', 'strike', 'put_iv']].copy()
        puts['right'] = 'P'
        puts = puts.rename(columns={'put_iv': 'actual_iv'})

        actual_df = pd.concat([calls, puts], ignore_index=True)
    else:
        actual_df = pd.DataFrame()

    return predicted_df, actual_df

# --- Chargement des données ---
predicted_df, actual_df = load_data()

if predicted_df.empty:
    st.warning("⚠️ Aucune donnée de prédiction à afficher.")
else:
    st.caption("🧠 Prédictions :")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Predicted IV - Calls")
        df_call = predicted_df[predicted_df["right"] == "C"]
        if not df_call.empty:
            fig = px.line(df_call, x="strike", y="predicted_iv", title="Calls")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Pas de prédictions Call disponibles.")

    with col2:
        st.subheader("📉 Predicted IV - Puts")
        df_put = predicted_df[predicted_df["right"] == "P"]
        if not df_put.empty:
            fig = px.line(df_put, x="strike", y="predicted_iv", title="Puts")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Pas de prédictions Put disponibles.")

    # Affichage de l’actual IV en dessous
    st.markdown("---")
    st.caption("📊 IV Réel (marché actuel) :")

    col3, col4 = st.columns(2)

    if not actual_df.empty and "right" in actual_df.columns:
        with col3:
            df_call_actual = actual_df[actual_df["right"] == "C"]
            if not df_call_actual.empty:
                st.subheader("📈 Actual IV - Calls")
                fig = px.line(df_call_actual, x="strike", y="actual_iv", title="Calls (actual)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Pas de IV réel Call disponible.")

        with col4:
            df_put_actual = actual_df[actual_df["right"] == "P"]
            if not df_put_actual.empty:
                st.subheader("📉 Actual IV - Puts")
                fig = px.line(df_put_actual, x="strike", y="actual_iv", title="Puts (actual)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Pas de IV réel Put disponible.")
    else:
        st.warning("⚠️ Aucune donnée IV réelle trouvée pour la maturité la plus récente.")
