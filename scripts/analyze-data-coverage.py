"""
Analyse-Skript zur Übersicht der vorhandenen Städte und Indikatoren
in der DuckDB-Datenbank.
"""

import duckdb
from pathlib import Path

# Pfad zur Datenbank
DB_PATH = Path(__file__).parent.parent / "data" / "db" / "data.duckdb"


def main():
    con = duckdb.connect(str(DB_PATH), read_only=True)

    print("=" * 60)
    print("DATENBANK-ANALYSE:  Städte und Indikatoren")
    print("=" * 60)

    # 1. Übersicht der Tabellen
    print("\n📊 TABELLEN IN DER DATENBANK:")
    print("-" * 40)
    tables = con.execute("SHOW TABLES").fetchall()
    for table in tables:
        count = con.execute(f"SELECT COUNT(*) FROM {table[0]}").fetchone()[0]
        print(f"  • {table[0]}: {count: ,} Einträge")

    # 2. Städte-Analyse
    print("\n\️ STÄDTE-ÜBERSICHT:")
    print("-" * 40)

    try:
        # Versuche verschiedene mögliche Spaltenstrukturen
        cities_query = """
            SELECT DISTINCT 
                geo_code,
                geo_name
            FROM dim_geo
            WHERE geo_code LIKE '%C' OR geo_code LIKE '%K'  -- City codes enden oft auf C oder K
            ORDER BY geo_code
        """
        cities_df = con.execute(cities_query).df()

        if len(cities_df) == 0:
            # Alternative: Alle geo-Einträge anzeigen
            cities_df = con.execute("SELECT * FROM dim_geo LIMIT 50").df()
            print("  (Zeige erste 50 Geo-Einträge)")

        print(f"  Anzahl Städte/Regionen: {len(cities_df)}")

        # Gruppiere nach Land (erste 2 Zeichen des Codes)
        if 'geo_code' in cities_df.columns:
            cities_df['country'] = cities_df['geo_code'].str[:2]
            country_counts = cities_df.groupby('country').size().sort_values(ascending=False)

            print("\n  Städte pro Land:")
            for country, count in country_counts.head(15).items():
                print(f"    {country}: {count} Städte")
            if len(country_counts) > 15:
                print(f"    ...  und {len(country_counts) - 15} weitere Länder")

    except Exception as e:
        print(f"  Fehler bei Städte-Analyse: {e}")
        # Zeige Schema von dim_geo
        schema = con.execute("DESCRIBE dim_geo").fetchall()
        print("  Schema von dim_geo:", schema)

    # 3. Indikatoren-Analyse
    print("\n INDIKATOREN-ÜBERSICHT:")
    print("-" * 40)

    try:
        indicators = con.execute("""
            SELECT indicator_code, indicator_name, source
            FROM dim_indicator
            ORDER BY source, indicator_code
        """).df()

        print(f"  Anzahl Indikatoren: {len(indicators)}")

        # Gruppiere nach Quelle
        if 'source' in indicators.columns:
            source_counts = indicators.groupby('source').size()
            print("\n  Indikatoren pro Quelle:")
            for source, count in source_counts.items():
                print(f"    {source}: {count} Indikatoren")

        print("\n  Liste aller Indikatoren:")
        for _, row in indicators.iterrows():
            name = row.get('indicator_name', row.get('indicator_code', 'N/A'))
            print(f"    • {name[: 60]}...")

    except Exception as e:
        print(f"  Fehler bei Indikatoren-Analyse: {e}")

    # 4. Datenvollständigkeit pro Stadt
    print("\n\n📋 DATENVOLLSTÄNDIGKEIT:")
    print("-" * 40)

    try:
        completeness = con.execute("""
            SELECT 
                g.geo_code,
                g.geo_name,
                COUNT(DISTINCT f.indicator_id) as num_indicators,
                COUNT(*) as num_datapoints,
                MIN(t.year) as min_year,
                MAX(t.year) as max_year
            FROM fact_measurements f
            JOIN dim_geo g ON f.geo_id = g.geo_id
            JOIN dim_time t ON f.time_id = t.time_id
            WHERE g.geo_code LIKE '%C' OR g.geo_code LIKE '%K'
            GROUP BY g.geo_code, g. geo_name
            ORDER BY num_indicators DESC
            LIMIT 20
        """).df()

        print("  Top 20 Städte nach Datenvollständigkeit:")
        print(completeness.to_string(index=False))

    except Exception as e:
        print(f"  Fehler bei Vollständigkeits-Analyse:  {e}")

    # 5. Clustering-Empfehlung
    print("\n\n EMPFEHLUNG FÜR CLUSTERING:")
    print("-" * 40)

    try:
        # Finde Städte mit den meisten gemeinsamen Indikatoren
        cluster_candidates = con.execute("""
            WITH city_indicators AS (
                SELECT 
                    g.geo_code,
                    g.geo_name,
                    COUNT(DISTINCT f.indicator_id) as num_indicators
                FROM fact_measurements f
                JOIN dim_geo g ON f. geo_id = g.geo_id
                WHERE g.geo_code LIKE '%C'
                GROUP BY g.geo_code, g.geo_name
            )
            SELECT 
                num_indicators,
                COUNT(*) as num_cities
            FROM city_indicators
            GROUP BY num_indicators
            ORDER BY num_indicators DESC
        """).df()

        print("  Verteilung der Indikator-Abdeckung:")
        print(cluster_candidates.to_string(index=False))

        # Empfehlung
        if len(cluster_candidates) > 0:
            max_indicators = cluster_candidates['num_indicators'].max()
            recommended = con.execute(f"""
                SELECT g.geo_code, g.geo_name
                FROM fact_measurements f
                JOIN dim_geo g ON f.geo_id = g.geo_id
                WHERE g.geo_code LIKE '%C'
                GROUP BY g.geo_code, g.geo_name
                HAVING COUNT(DISTINCT f.indicator_id) >= {max_indicators * 0.7}
            """).df()

            print("\n  Empfohlene Städte für Clustering (min.  70% Abdeckung):")
            print(f"  Anzahl:  {len(recommended)} Städte")

    except Exception as e:
        print(f"  Fehler bei Clustering-Empfehlung: {e}")

    # 6. Export für weitere Analyse
    print("\n\n💾 EXPORT:")
    print("-" * 40)

    try:
        # Erstelle eine Pivot-Tabelle für Clustering
        pivot_query = """
            SELECT 
                g.geo_code,
                g.geo_name,
                i.indicator_code,
                AVG(f.value) as avg_value
            FROM fact_measurements f
            JOIN dim_geo g ON f. geo_id = g.geo_id
            JOIN dim_indicator i ON f.indicator_id = i.indicator_id
            JOIN dim_time t ON f.time_id = t.time_id
            WHERE g.geo_code LIKE '%C'
            AND t.year >= 2018  -- Neueste Daten
            GROUP BY g.geo_code, g. geo_name, i.indicator_code
        """

        pivot_df = con.execute(pivot_query).df()

        if len(pivot_df) > 0:
            # Pivotiere die Daten
            pivot_table = pivot_df.pivot_table(
                index=['geo_code', 'geo_name'],
                columns='indicator_code',
                values='avg_value'
            ).reset_index()

            output_path = Path(__file__).parent.parent / "output" / "clustering_data.csv"
            pivot_table.to_csv(output_path, index=False)
            print(f"  ✅ Clustering-Daten exportiert nach:  {output_path}")
            print(f"     Städte: {len(pivot_table)}, Indikatoren: {len(pivot_table.columns) - 2}")

    except Exception as e:
        print(f"  Fehler beim Export: {e}")

    con.close()
    print("\n" + "=" * 60)
    print("Analyse abgeschlossen!")
    print("=" * 60)


if __name__ == "__main__":
    main()