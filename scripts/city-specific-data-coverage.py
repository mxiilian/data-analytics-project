"""
Analyse-Skript zur Übersicht der vorhandenen Städte und Indikatoren
in der DuckDB-Datenbank.

Nutzung:
    python scripts/analyze_data_coverage.py                     # Zeigt Gesamtübersicht
    python scripts/analyze_data_coverage.py Berlin Hamburg      # Zeigt Details für bestimmte Städte
    python scripts/analyze_data_coverage.py --list              # Listet alle verfügbaren Städte
"""

import duckdb
import pandas as pd
from pathlib import Path
import sys

# Pfad zur Datenbank
DB_PATH = Path(__file__).parent.parent / "data" / "db" / "data.duckdb"


def get_connection():
    """Erstellt eine Verbindung zur Datenbank."""
    return duckdb.connect(str(DB_PATH), read_only=True)


def is_city_code(geo_code):
    """Prüft ob ein geo_code eine Stadt ist (endet auf C oder K) oder ein Land (2 Buchstaben)."""
    if not geo_code:
        return False
    # Städte haben meist Codes wie DE001C, FR001K etc.
    # Länder haben 2-Buchstaben-Codes wie DE, FR, AT
    return len(geo_code) > 2


def list_all_cities(con):
    """Listet alle verfügbaren Städte in der Datenbank."""
    print("\n🏙️ ALLE VERFÜGBAREN STÄDTE:")
    print("=" * 60)

    try:
        cities_df = con.execute("""
            SELECT DISTINCT 
                geo_code,
                geo_name,
                country_code
            FROM dim_geo
            WHERE LENGTH(geo_code) > 2
            ORDER BY country_code, geo_name
        """).df()

        # Gruppiere nach Land
        for country in sorted(cities_df['country_code'].unique()):
            country_cities = cities_df[cities_df['country_code'] == country]
            print(f"\n  {country}:")
            for _, row in country_cities.iterrows():
                print(f"    • {row['geo_name']} ({row['geo_code']})")

        print(f"\n  Gesamt: {len(cities_df)} Städte")

    except Exception as e:
        print(f"  Fehler: {e}")


def get_indicator_availability(con, indicator_code):
    """
    Prüft für welche Geo-Typen ein Indikator verfügbar ist.
    Gibt zurück: 'cities', 'countries', 'both', oder 'none'
    """
    result = con.execute(f"""
        SELECT 
            SUM(CASE WHEN LENGTH(geo_code) > 2 THEN 1 ELSE 0 END) as city_count,
            SUM(CASE WHEN LENGTH(geo_code) = 2 THEN 1 ELSE 0 END) as country_count
        FROM fact_measurements
        WHERE indicator_code = '{indicator_code}'
    """).fetchone()

    city_count = result[0] or 0
    country_count = result[1] or 0

    if city_count > 0 and country_count > 0:
        return 'both', city_count, country_count
    elif city_count > 0:
        return 'cities', city_count, 0
    elif country_count > 0:
        return 'countries', 0, country_count
    else:
        return 'none', 0, 0


def analyze_city(con, city_name):
    """Analysiert die Datenabdeckung für eine bestimmte Stadt."""
    print(f"\n{'=' * 60}")
    print(f"📊 DATENABDECKUNG FÜR:  {city_name. upper()}")
    print("=" * 60)

    # Suche Stadt (case-insensitive, partial match)
    try:
        city_info = con.execute(f"""
            SELECT geo_code, geo_name, country_code
            FROM dim_geo
            WHERE LOWER(geo_name) LIKE LOWER('%{city_name}%')
            AND LENGTH(geo_code) > 2
            LIMIT 5
        """).df()

        if len(city_info) == 0:
            print(f"  ❌ Keine Stadt gefunden mit '{city_name}'")
            print("  Tipp: Nutze --list um alle verfügbaren Städte zu sehen")
            return

        if len(city_info) > 1:
            print(f"  ⚠️ Mehrere Treffer gefunden:")
            for _, row in city_info. iterrows():
                print(f"    • {row['geo_name']} ({row['geo_code']})")
            print(f"  Verwende ersten Treffer: {city_info.iloc[0]['geo_name']}")

        city = city_info.iloc[0]
        geo_code = city['geo_code']
        print(f"\n  Stadt: {city['geo_name']}")
        print(f"  Code: {geo_code}")
        print(f"  Land:  {city['country_code']}")

        # Anzahl Datenpunkte
        stats = con.execute(f"""
            SELECT 
                COUNT(*) as total_datapoints,
                COUNT(DISTINCT indicator_code) as num_indicators,
                COUNT(DISTINCT year) as num_years
            FROM fact_measurements
            WHERE geo_code = '{geo_code}'
        """).fetchone()

        print(f"\n  📈 Statistiken:")
        print(f"    • Datenpunkte gesamt: {stats[0]: ,}")
        print(f"    • Anzahl Indikatoren: {stats[1]}")
        print(f"    • Anzahl Jahre: {stats[2]}")

        # Zeitraum
        time_range = con. execute(f"""
            SELECT MIN(year), MAX(year)
            FROM fact_measurements
            WHERE geo_code = '{geo_code}'
        """).fetchone()

        if time_range[0]:
            print(f"    • Zeitraum: {time_range[0]} - {time_range[1]}")

        # Indikatoren-Details
        print(f"\n  📋 Verfügbare Indikatoren:")
        indicators = con.execute(f"""
            SELECT 
                f.indicator_code,
                i.indicator_name,
                i.dataset_name,
                COUNT(*) as datapoints,
                MIN(f.year) as von,
                MAX(f.year) as bis
            FROM fact_measurements f
            LEFT JOIN dim_indicator i ON f.indicator_code = i.indicator_code
            WHERE f.geo_code = '{geo_code}'
            GROUP BY f.indicator_code, i.indicator_name, i.dataset_name
            ORDER BY i.dataset_name, i.indicator_name
        """).df()

        if len(indicators) > 0:
            current_dataset = None
            for _, row in indicators.iterrows():
                dataset = row['dataset_name'] if row['dataset_name'] else 'Unbekannt'
                if dataset != current_dataset:
                    current_dataset = dataset
                    print(f"\n    [{dataset}]")

                name = row['indicator_name'][: 90] if row['indicator_name'] else row['indicator_code']
                print(f"      ✓ {name}")
                print(f"        ({row['datapoints']} Einträge, {row['von']}-{row['bis']})")
        else:
            print("    Keine Indikatoren gefunden")

        # Fehlende Indikatoren - mit Prüfung ob für Städte verfügbar
        print(f"\n  ❌ Fehlende Indikatoren:")
        missing = con.execute(f"""
            SELECT indicator_code, indicator_name, dataset_name
            FROM dim_indicator
            WHERE indicator_code NOT IN (
                SELECT DISTINCT indicator_code 
                FROM fact_measurements 
                WHERE geo_code = '{geo_code}'
            )
            ORDER BY dataset_name, indicator_name
        """).df()

        if len(missing) > 0:
            missing_for_cities = []  # Fehlt, aber für andere Städte verfügbar
            only_countries = []      # Nur für Länder verfügbar

            for _, row in missing. iterrows():
                availability, city_count, country_count = get_indicator_availability(con, row['indicator_code'])

                indicator_info = {
                    'indicator_code': row['indicator_code'],
                    'indicator_name': row['indicator_name'],
                    'dataset_name': row['dataset_name'],
                    'availability': availability,
                    'city_count': city_count,
                    'country_count': country_count
                }

                if availability in ['cities', 'both']:
                    missing_for_cities.append(indicator_info)
                else:
                    only_countries.append(indicator_info)

            # Zeige fehlende Indikatoren die für Städte verfügbar sind
            if missing_for_cities:
                print(f"\n    🏙️ Fehlt für diese Stadt (aber für andere Städte verfügbar):")
                current_dataset = None
                for ind in missing_for_cities:
                    dataset = ind['dataset_name'] if ind['dataset_name'] else 'Unbekannt'
                    if dataset != current_dataset:
                        current_dataset = dataset
                        print(f"\n      [{dataset}]")

                    name = ind['indicator_name'][:45] if ind['indicator_name'] else ind['indicator_code']
                    print(f"        ✗ {name}")
                    print(f"          (verfügbar für {ind['city_count']} andere Städte)")

            # Zeige Indikatoren die nur für Länder verfügbar sind
            if only_countries:
                print(f"\n    🌍 Nur auf Länderebene verfügbar (nicht für Städte):")
                current_dataset = None
                for ind in only_countries:
                    dataset = ind['dataset_name'] if ind['dataset_name'] else 'Unbekannt'
                    if dataset != current_dataset:
                        current_dataset = dataset
                        print(f"\n      [{dataset}]")

                    name = ind['indicator_name'][: 45] if ind['indicator_name'] else ind['indicator_code']
                    print(f"        ○ {name}")
                    if ind['country_count'] > 0:
                        print(f"          (verfügbar für {ind['country_count']} Länder)")

            # Zusammenfassung
            print(f"\n    📊 Zusammenfassung fehlende Indikatoren:")
            print(f"       • Für andere Städte verfügbar: {len(missing_for_cities)}")
            print(f"       • Nur auf Länderebene:  {len(only_countries)}")
        else:
            print("    Alle Indikatoren vorhanden!  ✅")

    except Exception as e:
        print(f"  Fehler bei der Analyse: {e}")
        import traceback
        traceback.print_exc()


def compare_cities(con, city_names):
    """Vergleicht die Datenabdeckung mehrerer Städte."""
    print(f"\n{'=' * 60}")
    print(f"📊 VERGLEICH DER DATENABDECKUNG")
    print("=" * 60)

    comparison_data = []
    geo_codes = []

    for city_name in city_names:
        try:
            city_info = con. execute(f"""
                SELECT geo_code, geo_name, country_code
                FROM dim_geo
                WHERE LOWER(geo_name) LIKE LOWER('%{city_name}%')
                AND LENGTH(geo_code) > 2
                LIMIT 1
            """).df()

            if len(city_info) == 0:
                print(f"  ⚠️ '{city_name}' nicht gefunden")
                continue

            city = city_info. iloc[0]
            geo_code = city['geo_code']
            geo_codes.append(geo_code)

            stats = con.execute(f"""
                SELECT 
                    COUNT(*) as total_datapoints,
                    COUNT(DISTINCT indicator_code) as num_indicators,
                    MIN(year) as min_year,
                    MAX(year) as max_year
                FROM fact_measurements
                WHERE geo_code = '{geo_code}'
            """).fetchone()

            comparison_data.append({
                'Stadt': city['geo_name'],
                'Code': geo_code,
                'Land': city['country_code'],
                'Datenpunkte': stats[0],
                'Indikatoren': stats[1],
                'Von': stats[2],
                'Bis': stats[3]
            })

        except Exception as e:
            print(f"  Fehler bei '{city_name}':  {e}")

    if comparison_data:
        df = pd.DataFrame(comparison_data)
        print("\n" + df.to_string(index=False))

        # Gemeinsame Indikatoren finden
        if len(geo_codes) > 1:
            print(f"\n  🔗 Gemeinsame Indikatoren:")

            geo_codes_str = "', '".join(geo_codes)
            common_indicators = con.execute(f"""
                SELECT i.indicator_code, i.indicator_name, i.dataset_name
                FROM dim_indicator i
                WHERE i.indicator_code IN (
                    SELECT indicator_code
                    FROM fact_measurements
                    WHERE geo_code IN ('{geo_codes_str}')
                    GROUP BY indicator_code
                    HAVING COUNT(DISTINCT geo_code) = {len(geo_codes)}
                )
                ORDER BY i. dataset_name, i.indicator_name
            """).df()

            print(f"    Anzahl:  {len(common_indicators)}")
            current_dataset = None
            for _, row in common_indicators.iterrows():
                dataset = row['dataset_name'] if row['dataset_name'] else 'Unbekannt'
                if dataset != current_dataset:
                    current_dataset = dataset
                    print(f"\n    [{dataset}]")

                name = row['indicator_name'][: 60] if row['indicator_name'] else row['indicator_code']
                print(f"      • {name}")


def show_overview(con):
    """Zeigt eine Gesamtübersicht der Datenbank."""
    print("=" * 60)
    print("DATENBANK-ANALYSE:  Übersicht")
    print("=" * 60)

    # Tabellen
    print("\n📊 TABELLEN:")
    print("-" * 40)
    tables = con.execute("SHOW TABLES").fetchall()
    for table in tables:
        count = con.execute(f"SELECT COUNT(*) FROM {table[0]}").fetchone()[0]
        print(f"  • {table[0]}: {count: ,} Einträge")

    # Anzahl Städte und Länder
    print("\n\n🌍 GEO-ÜBERSICHT:")
    print("-" * 40)
    geo_stats = con.execute("""
        SELECT 
            SUM(CASE WHEN LENGTH(geo_code) > 2 THEN 1 ELSE 0 END) as cities,
            SUM(CASE WHEN LENGTH(geo_code) = 2 THEN 1 ELSE 0 END) as countries
        FROM dim_geo
    """).fetchone()
    print(f"  • Städte: {geo_stats[0]}")
    print(f"  • Länder: {geo_stats[1]}")

    # Indikatoren nach Verfügbarkeit
    print("\n\n📈 INDIKATOREN NACH VERFÜGBARKEIT:")
    print("-" * 40)

    indicator_stats = con.execute("""
        WITH indicator_geo_types AS (
            SELECT 
                indicator_code,
                MAX(CASE WHEN LENGTH(geo_code) > 2 THEN 1 ELSE 0 END) as has_cities,
                MAX(CASE WHEN LENGTH(geo_code) = 2 THEN 1 ELSE 0 END) as has_countries
            FROM fact_measurements
            GROUP BY indicator_code
        )
        SELECT 
            SUM(CASE WHEN has_cities = 1 AND has_countries = 1 THEN 1 ELSE 0 END) as both,
            SUM(CASE WHEN has_cities = 1 AND has_countries = 0 THEN 1 ELSE 0 END) as cities_only,
            SUM(CASE WHEN has_cities = 0 AND has_countries = 1 THEN 1 ELSE 0 END) as countries_only
        FROM indicator_geo_types
    """).fetchone()

    print(f"  • Für Städte UND Länder:  {indicator_stats[0]}")
    print(f"  • Nur für Städte:  {indicator_stats[1]}")
    print(f"  • Nur für Länder: {indicator_stats[2]}")

    # Top 10 Städte nach Datenvollständigkeit
    print("\n\n🏆 TOP 10 STÄDTE (nach Datenabdeckung):")
    print("-" * 40)

    try:
        top_cities = con. execute("""
            SELECT 
                g.geo_name as Stadt,
                g. geo_code as Code,
                g. country_code as Land,
                COUNT(DISTINCT f. indicator_code) as Indikatoren,
                COUNT(*) as Datenpunkte
            FROM fact_measurements f
            JOIN dim_geo g ON f.geo_code = g.geo_code
            WHERE LENGTH(g.geo_code) > 2
            GROUP BY g.geo_code, g.geo_name, g. country_code
            ORDER BY Indikatoren DESC, Datenpunkte DESC
            LIMIT 10
        """).df()

        print(top_cities.to_string(index=False))

    except Exception as e:
        print(f"  Fehler:  {e}")

    # Indikatoren nach Dataset
    print("\n\n📁 INDIKATOREN NACH DATASET:")
    print("-" * 40)
    try:
        datasets = con.execute("""
            SELECT 
                dataset_name,
                COUNT(*) as anzahl
            FROM dim_indicator
            GROUP BY dataset_name
            ORDER BY anzahl DESC
        """).df()

        for _, row in datasets.iterrows():
            name = row['dataset_name'] if row['dataset_name'] else 'Unbekannt'
            print(f"  • {name}:  {row['anzahl']} Indikatoren")

    except Exception as e:
        print(f"  Fehler: {e}")

    print("\n\n💡 NUTZUNG:")
    print("-" * 40)
    print("  python scripts/analyze_data_coverage.py Berlin Hamburg  # Vergleiche Städte")
    print("  python scripts/analyze_data_coverage.py --list          # Liste alle Städte")


def main():
    con = get_connection()

    args = sys.argv[1:]

    if not args:
        # Keine Argumente:  Zeige Übersicht
        show_overview(con)
    elif args[0] == '--list':
        # Liste alle Städte
        list_all_cities(con)
    elif len(args) == 1:
        # Eine Stadt:  Detailanalyse
        analyze_city(con, args[0])
    else:
        # Mehrere Städte:  Vergleich + Einzelanalysen
        compare_cities(con, args)
        for city in args:
            analyze_city(con, city)

    con. close()


if __name__ == "__main__":
    main()