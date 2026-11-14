import hashlib
import pathlib
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from source_io import fetch

SOURCE_URL = 'https://data.www.sbir.gov/mod_awarddatapublic_no_abstract/award_data_no_abstract.csv'
SOURCE_NAME = 'sbir_awards_no_abstract.csv'
TARGET = 'phase2_conversion'
ID_COL = 'record_id'
TIME_COL = 'award_date'
FEATURES = [
    'agency', 'branch', 'program', 'award_year', 'award_amount', 'topic_code',
    'state', 'number_employees', 'hubzone_owned', 'women_owned',
    'socially_economically_disadvantaged',
]
CSV_COLUMNS = [ID_COL, TIME_COL] + [c for c in FEATURES if c != TIME_COL] + [TARGET]
ALIASES = {
    'award_year': ['award_year', 'awardyear', 'fiscal_year', 'year'],
    'agency': ['agency', 'awarding_agency', 'agency_name'],
    'branch': ['branch', 'agency_branch', 'sub_agency', 'subagency', 'department'],
    'phase': ['phase', 'award_phase', 'phase_code'],
    'program': ['program', 'program_name', 'award_program'],
    'award_amount': ['award_amount', 'award_amount_usd', 'amount', 'award_value'],
    'topic_code': ['topic_code', 'topic', 'topic_number', 'solicitation_topic'],
    'state': ['state', 'state_code', 'awardee_state', 'firm_state'],
    'number_employees': ['number_employees', 'number_of_employees', 'employees', 'employee_count'],
    'hubzone_owned': ['hubzone_owned', 'hub_zone_owned', 'hubzone'],
    'women_owned': ['women_owned', 'woman_owned', 'womenowned'],
    'socially_economically_disadvantaged': ['socially_economically_disadvantaged', 'socially_and_economically_disadvantaged', 'minority_owned', 'sed'],
    'uei': ['uei', 'unique_entity_id', 'unique_entity_identifier'],
    'duns': ['duns', 'duns_number', 'duns_no'],
    'company': ['company', 'firm', 'company_name', 'organization', 'awardee', 'awardee_name', 'business_name'],
    'proposal_award_date': ['proposal_award_date', 'award_date', 'proposal_date', 'date_of_award'],
}
REQUIRED = [
    'award_year', 'agency', 'branch', 'phase', 'program', 'award_amount', 'topic_code',
    'state', 'number_employees', 'hubzone_owned', 'women_owned',
    'socially_economically_disadvantaged',
]
LEGAL = set('inc incorporated llc corp corporation co company ltd limited lp llp plc pc pa the'.split())
USPS = set(
    'AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ '
    'NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC AS GU MP PR VI MH FM PW AA AE AP'.split()
)
STATE_NAME_TO_CODE = {
    'ALABAMA': 'AL', 'ALASKA': 'AK', 'ARIZONA': 'AZ', 'ARKANSAS': 'AR', 'CALIFORNIA': 'CA',
    'COLORADO': 'CO', 'CONNECTICUT': 'CT', 'DELAWARE': 'DE', 'DISTRICT OF COLUMBIA': 'DC',
    'D C': 'DC', 'FLORIDA': 'FL', 'GEORGIA': 'GA', 'HAWAII': 'HI', 'IDAHO': 'ID',
    'ILLINOIS': 'IL', 'INDIANA': 'IN', 'IOWA': 'IA', 'KANSAS': 'KS', 'KENTUCKY': 'KY',
    'LOUISIANA': 'LA', 'MAINE': 'ME', 'MARYLAND': 'MD', 'MASSACHUSETTS': 'MA',
    'MICHIGAN': 'MI', 'MINNESOTA': 'MN', 'MISSISSIPPI': 'MS', 'MISSOURI': 'MO',
    'MONTANA': 'MT', 'NEBRASKA': 'NE', 'NEVADA': 'NV', 'NEW HAMPSHIRE': 'NH',
    'NEW JERSEY': 'NJ', 'NEW MEXICO': 'NM', 'NEW YORK': 'NY', 'NORTH CAROLINA': 'NC',
    'NORTH DAKOTA': 'ND', 'OHIO': 'OH', 'OKLAHOMA': 'OK', 'OREGON': 'OR',
    'PENNSYLVANIA': 'PA', 'RHODE ISLAND': 'RI', 'SOUTH CAROLINA': 'SC',
    'SOUTH DAKOTA': 'SD', 'TENNESSEE': 'TN', 'TEXAS': 'TX', 'UTAH': 'UT', 'VERMONT': 'VT',
    'VIRGINIA': 'VA', 'WASHINGTON': 'WA', 'WEST VIRGINIA': 'WV', 'WISCONSIN': 'WI',
    'WYOMING': 'WY',
    'PUERTO RICO': 'PR', 'GUAM': 'GU', 'AMERICAN SAMOA': 'AS',
    'VIRGIN ISLANDS': 'VI', 'U S VIRGIN ISLANDS': 'VI',
    'UNITED STATES VIRGIN ISLANDS': 'VI',
    'NORTHERN MARIANA ISLANDS': 'MP', 'MARIANA ISLANDS': 'MP',
    'MARSHALL ISLANDS': 'MH', 'FEDERATED STATES OF MICRONESIA': 'FM',
    'MICRONESIA': 'FM', 'PALAU': 'PW',
}


def _clean(value):
    if value is None:
        return ''
    try:
        if pd.isna(value):
            return ''
    except Exception:
        pass
    text = str(value).strip()
    return '' if text.lower() in ('', 'nan', 'none', 'null', 'na', 'n/a') else text


def _norm_header(text):
    return ''.join(ch.lower() if ch.isalnum() else '_' for ch in str(text)).strip('_')


def _header_map(observed):
    norm_to_actual = {}
    for col in observed:
        key = _norm_header(col)
        if key and key not in norm_to_actual:
            norm_to_actual[key] = col
    mapping = {}
    used = set()
    for canon, aliases in ALIASES.items():
        for alias in aliases:
            actual = norm_to_actual.get(alias)
            if actual is not None and actual not in used:
                mapping[canon] = actual
                used.add(actual)
                break
    return mapping


def _fail(message):
    raise AssertionError(str(message)[:4000])


def _check(checks, name, observed, expected, passed, detail=None, kind=None):
    ok = bool(passed)
    record = {'check': name, 'observed': observed, 'expected': expected, 'passed': ok}
    if kind:
        record['kind'] = kind
    checks.append(record)
    if not ok:
        message = 'Source check failed: ' + name + '. Observed: ' + str(observed) + '. Expected: ' + str(expected)
        if detail:
            message += '. Detail: ' + str(detail)
        _fail(message)


def _norm_name(value):
    text = _clean(value).lower()
    if not text:
        return ''
    text = ''.join(ch if ch.isalnum() else ' ' for ch in text)
    return ' '.join(tok for tok in text.split() if tok and tok not in LEGAL)


def _compact(value):
    text = _clean(value).upper().replace('.', ' ').replace(',', ' ')
    return ' '.join(text.split())


def _state_text_like(key):
    if len(key) < 2:
        return False
    return all(ch.isalpha() or ch == ' ' for ch in key)


def _state_kind(value):
    key = _compact(value)
    if not key:
        return 'blank'
    if len(key) == 2 and key.isalpha() and key in USPS:
        return 'code'
    if key in STATE_NAME_TO_CODE:
        return 'name'
    return 'other'


def _state_code(value):
    key = _compact(value)
    if not key:
        return ''
    if len(key) == 2 and key.isalpha() and key in USPS:
        return key
    if key in STATE_NAME_TO_CODE:
        return STATE_NAME_TO_CODE[key]
    return key


def _phase_series(series):
    text = series.map(_clean).str.upper()
    text = text.map(lambda value: ''.join(ch for ch in value if ch.isalnum()))
    return text.map({
        '1': 1, 'I': 1, 'PHASEI': 1, 'PHASE1': 1,
        '2': 2, 'II': 2, 'PHASEII': 2, 'PHASE2': 2,
        '3': 3, 'III': 3, 'PHASEIII': 3, 'PHASE3': 3,
    })


def _numeric_series(series, money=False):
    text = series.map(_clean)
    text = text.map(lambda value: value.replace('$', '').replace(',', '').replace(' ', ''))
    return pd.to_numeric(text, errors='coerce')


def _ynu_series(series):
    text = series.map(_clean).str.upper()
    out = text.map({
        'Y': 'Y', 'YES': 'Y', 'TRUE': 'Y', '1': 'Y',
        'N': 'N', 'NO': 'N', 'FALSE': 'N', '0': 'N',
        'U': 'U', 'UNKNOWN': 'U',
    })
    out = out.where(out.notna(), text)
    return out.where(text.ne(''), np.nan)


def _firm_keys(frame):
    uei = frame['uei'].map(_clean).str.upper().map(lambda value: ''.join(ch for ch in value if ch.isalnum()))
    duns = frame['duns'].map(_clean).map(lambda value: ''.join(ch for ch in value if ch.isdigit()))
    company = frame['company'].map(_norm_name)
    state = frame['state'].map(_clean).str.upper()
    keys = pd.Series(np.nan, index=frame.index, dtype=object)
    mask = uei.ne('')
    keys[mask] = 'uei:' + uei[mask]
    mask = keys.isna() & duns.ne('')
    keys[mask] = 'duns:' + duns[mask]
    state_ok = state.map(lambda value: len(value) >= 2)
    mask = keys.isna() & company.ne('') & state_ok
    keys[mask] = 'name:' + company[mask] + '|' + state[mask]
    return keys


def _parse_datetime(value):
    text = _clean(value)
    if not text:
        return None
    parsed = pd.to_datetime(text, errors='coerce', utc=True)
    if pd.isna(parsed):
        return None
    return parsed.tz_convert(None)


def _award_date(year, value):
    year = int(year)
    dt = _parse_datetime(value)
    start = pd.Timestamp(year=year - 1, month=10, day=1)
    end = pd.Timestamp(year=year, month=10, day=1)
    if dt is not None and start <= dt < end:
        return dt
    return pd.Timestamp(year=year, month=9, day=30)


def _chunks(path, mapping):
    reverse = {actual: canon for canon, actual in mapping.items()}
    for chunk in pd.read_csv(
        path, dtype=str, low_memory=False, chunksize=200000,
        encoding='utf-8-sig', keep_default_na=False, na_values=[''],
    ):
        chunk = chunk.rename(columns=reverse)
        for name in ('uei', 'duns', 'company', 'proposal_award_date'):
            if name not in chunk.columns:
                chunk[name] = ''
        if 'state' not in chunk.columns:
            chunk['state'] = ''
        chunk['state_raw'] = chunk['state'].map(_clean)
        chunk['state'] = chunk['state_raw'].map(_state_code)
        yield chunk


def _split_for_year(year):
    if 2005 <= year <= 2014:
        return 'train'
    if 2015 <= year <= 2016:
        return 'eval'
    if 2017 <= year <= 2018:
        return 'holdout'
    return None


def build(output: pathlib.Path, cache: pathlib.Path):
    output = pathlib.Path(output)
    cache = pathlib.Path(cache)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'public').mkdir(parents=True, exist_ok=True)
    (output / 'private').mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    checks = []
    path = pathlib.Path(fetch(SOURCE_URL, cache, SOURCE_NAME))
    size = path.stat().st_size
    _check(
        checks,
        'source file present',
        {'name': path.name, 'bytes': int(size), 'source_member': SOURCE_NAME},
        'fetch returns a nonempty local file for the published CSV member',
        size > 0,
        detail='Selected source member ' + SOURCE_NAME + ' returned an empty file',
    )

    header = pd.read_csv(path, nrows=0, encoding='utf-8-sig')
    observed_headers = list(header.columns)
    mapping = _header_map(observed_headers)
    missing = [name for name in REQUIRED if name not in mapping]
    if not any(name in mapping for name in ('uei', 'duns', 'company')):
        missing.append('a firm identifier among uei, duns and company')
    _check(
        checks,
        'required source fields resolved',
        {
            'source_member': SOURCE_NAME,
            'observed_headers': observed_headers[:80],
            'resolved_names': sorted(mapping.keys()),
        },
        'every declared feature plus phase, firm identifier and agency maps to a published column',
        not missing,
        detail=(
            'Selected archive member ' + SOURCE_NAME
            + '. Observed headers: ' + repr(observed_headers[:80])
            + '. Missing required names: ' + repr(missing)
        ),
    )

    firm_min = {}
    firm_missing_year = set()
    phase2 = defaultdict(set)
    phase2_firm_ambiguous = set()
    unknown_phase = defaultdict(set)
    unknown_phase_firm_ambiguous = set()
    max_year = None
    raw_rows = 0
    missing_firm_key = 0
    phase_counts = Counter()
    program_counts = Counter()
    raw_state_counts = Counter()
    key_kinds = Counter()

    for chunk in _chunks(path, mapping):
        for raw_value, raw_count in chunk['state_raw'].value_counts().items():
            raw_state_counts[str(raw_value)] += int(raw_count)
        phases = _phase_series(chunk['phase'])
        years = _numeric_series(chunk['award_year'])
        agencies = chunk['agency'].map(_clean).str.upper()
        programs = chunk['program'].map(_clean).str.upper()
        keys = _firm_keys(chunk)
        raw_phase = chunk['phase'].map(_clean)
        for i in range(len(chunk)):
            raw_rows += 1
            phase_value = phases.iloc[i]
            year_value = years.iloc[i]
            key = keys.iloc[i]
            agency = agencies.iloc[i]
            program = programs.iloc[i]
            if pd.notna(phase_value):
                phase_counts[str(int(phase_value))] += 1
            elif raw_phase.iloc[i]:
                phase_counts['unknown'] += 1
            else:
                phase_counts['missing'] += 1
            program_counts[program if program else 'missing'] += 1
            if not key or pd.isna(key):
                missing_firm_key += 1
                continue
            key_kinds[str(key).split(':', 1)[0]] += 1
            if pd.isna(year_value):
                firm_missing_year.add(key)
                continue
            year = int(year_value)
            if max_year is None or year > max_year:
                max_year = year
            phase_num = int(phase_value) if pd.notna(phase_value) else None
            previous = firm_min.get(key)
            if previous is None or year < previous[0]:
                firm_min[key] = (year, {phase_num})
            elif year == previous[0]:
                previous[1].add(phase_num)
            if phase_num == 2:
                if agency:
                    phase2[(key, agency)].add(year)
                else:
                    phase2_firm_ambiguous.add(key)
            elif phase_num not in (1, 2, 3):
                if agency:
                    unknown_phase[(key, agency)].add(year)
                else:
                    unknown_phase_firm_ambiguous.add(key)

    if max_year is None:
        _fail('No parseable award_year values in source member ' + SOURCE_NAME)

    _check(
        checks,
        'firm identifier resolution coverage',
        {
            'rows_without_firm_key': int(missing_firm_key),
            'keyed_rows_by_kind': {kind: int(count) for kind, count in key_kinds.items()},
        },
        'firm identity resolves through UEI, else DUNS, else normalized company name with a state',
        sum(key_kinds.values()) > 0,
        detail='Source member ' + SOURCE_NAME + ' produced no resolvable firm identifiers',
    )

    state_kind_totals = Counter()
    state_other = Counter()
    for raw_value, raw_count in raw_state_counts.items():
        kind = _state_kind(raw_value)
        state_kind_totals[kind] += raw_count
        if kind == 'other':
            state_other[_compact(raw_value)] += raw_count
    bad_state_text = {value: count for value, count in state_other.items() if not _state_text_like(value)}
    _check(
        checks,
        'state vocabulary in the published file',
        {
            'source_member': SOURCE_NAME,
            'distinct_state_values': int(len(raw_state_counts)),
            'blank': int(state_kind_totals.get('blank', 0)),
            'two_letter_code': int(state_kind_totals.get('code', 0)),
            'full_state_name': int(state_kind_totals.get('name', 0)),
            'other_location_text': {value: int(count) for value, count in list(state_other.items())[:10]},
            'other_location_total': int(sum(state_other.values())),
            'non_text_values': {value: int(count) for value, count in list(bad_state_text.items())[:10]},
        },
        'the published state column holds two-letter USPS codes or full state and territory names; full names are canonicalized to codes and non-US location text is preserved verbatim',
        not bad_state_text,
        detail=(
            'Selected archive member ' + SOURCE_NAME
            + '. State values in the source are neither blank, a two-letter code, a known state or territory name, nor alphabetic location text: '
            + repr(list(bad_state_text.items())[:20])
        ),
    )

    eligible = {}
    for key, value in firm_min.items():
        year, phases = value
        if key in firm_missing_year or key in phase2_firm_ambiguous or key in unknown_phase_firm_ambiguous:
            continue
        if phases != {1}:
            continue
        if year < 2005 or year > 2018:
            continue
        if year + 4 > max_year:
            continue
        eligible[key] = year
    if not eligible:
        _fail(
            'No eligible first-time Phase I firms after population rules. Observed phase counts '
            + str(dict(phase_counts)) + ' and maximum award year ' + str(max_year) + '.'
        )

    candidates = defaultdict(list)
    for chunk in _chunks(path, mapping):
        keys = _firm_keys(chunk)
        mask = keys.isin(eligible)
        if not mask.any():
            continue
        sub = chunk[mask].copy()
        sub_keys = keys[mask]
        phases = _phase_series(sub['phase'])
        years = _numeric_series(sub['award_year'])
        agencies = sub['agency'].map(_clean).str.upper()
        branches = sub['branch'].map(_clean)
        programs = sub['program'].map(_clean).str.upper()
        amounts = _numeric_series(sub['award_amount'], money=True)
        topics = sub['topic_code'].map(_clean)
        states = sub['state'].map(_clean).str.upper()
        employees = _numeric_series(sub['number_employees'])
        hubzones = _ynu_series(sub['hubzone_owned'])
        women = _ynu_series(sub['women_owned'])
        socials = _ynu_series(sub['socially_economically_disadvantaged'])
        dates = sub['proposal_award_date'].map(_clean)
        for i in range(len(sub)):
            key = sub_keys.iloc[i]
            year_value = years.iloc[i]
            if pd.isna(year_value) or int(year_value) != eligible.get(key):
                continue
            phase_value = phases.iloc[i]
            if pd.isna(phase_value) or int(phase_value) != 1:
                continue
            candidates[key].append({
                'firm_key': key,
                'agency': agencies.iloc[i],
                'branch': branches.iloc[i],
                'program': programs.iloc[i],
                'award_year': int(year_value),
                'award_amount': amounts.iloc[i],
                'topic_code': topics.iloc[i],
                'state': states.iloc[i],
                'number_employees': employees.iloc[i],
                'hubzone_owned': hubzones.iloc[i],
                'women_owned': women.iloc[i],
                'socially_economically_disadvantaged': socials.iloc[i],
                'proposal_award_date': dates.iloc[i],
            })

    rows = []
    excluded = Counter()
    for key, year in eligible.items():
        recs = candidates.get(key, [])
        if not recs:
            excluded['no_index_record'] += 1
            continue
        if len(recs) == 1:
            chosen = recs[0]
        else:
            parsed = [_parse_datetime(rec['proposal_award_date']) for rec in recs]
            if any(value is None for value in parsed):
                excluded['ambiguous_same_year_missing_date'] += 1
                continue
            earliest = min(parsed)
            winners = [rec for rec, value in zip(recs, parsed) if value == earliest]
            if len(winners) != 1:
                excluded['ambiguous_same_year_tied_date'] += 1
                continue
            chosen = winners[0]
        agency = chosen['agency']
        if not agency:
            excluded['missing_agency'] += 1
            continue
        ambiguous = False
        for future in range(year + 1, year + 5):
            if future in unknown_phase.get((key, agency), set()):
                ambiguous = True
                break
        if ambiguous:
            excluded['unknown_phase_in_window'] += 1
            continue
        phase2_years = phase2.get((key, agency), set())
        label = 1 if any(future in phase2_years for future in range(year + 1, year + 5)) else 0
        record_id = 'sbir-' + hashlib.sha1(str(key).encode('utf-8')).hexdigest()[:24]
        award_date = _award_date(year, chosen['proposal_award_date'])
        rows.append({
            ID_COL: record_id,
            TIME_COL: award_date.strftime('%Y-%m-%dT%H:%M:%S.000'),
            'agency': agency,
            'branch': chosen['branch'],
            'program': chosen['program'],
            'award_year': year,
            'award_amount': chosen['award_amount'],
            'topic_code': chosen['topic_code'],
            'state': chosen['state'],
            'number_employees': chosen['number_employees'],
            'hubzone_owned': chosen['hubzone_owned'],
            'women_owned': chosen['women_owned'],
            'socially_economically_disadvantaged': chosen['socially_economically_disadvantaged'],
            TARGET: label,
            '_firm_key': key,
        })

    if not rows:
        _fail('No rows constructed. Exclusions ' + str(dict(excluded)) + '.')

    frame = pd.DataFrame(rows)
    frame[ID_COL] = frame[ID_COL].astype(str)
    frame['award_year'] = frame['award_year'].astype(int)
    frame[TARGET] = frame[TARGET].astype(int)
    frame['award_amount'] = pd.to_numeric(frame['award_amount'], errors='coerce')
    frame['number_employees'] = pd.to_numeric(frame['number_employees'], errors='coerce')
    frame[TIME_COL] = pd.to_datetime(frame[TIME_COL], errors='coerce')
    if frame[TIME_COL].isna().any():
        _fail('Shipped award_date values could not be parsed. Observed rows ' + str(len(frame)) + '.')
    frame[TIME_COL] = frame[TIME_COL].dt.strftime('%Y-%m-%dT%H:%M:%S.000')

    program_counts_included = Counter(frame['program'].fillna('missing').astype(str).tolist())
    bad_program = {name: count for name, count in program_counts_included.items() if name not in ('SBIR', 'STTR')}
    _check(
        checks,
        'program values for included rows',
        {'counts': dict(program_counts_included)},
        'program is SBIR or STTR on every included Phase I index record',
        not bad_program,
        detail='Selected source member ' + SOURCE_NAME + '. Observed program counts: ' + repr(dict(program_counts_included)),
    )

    state_values = frame['state'].fillna('').astype(str)
    missing_state = int((state_values == '').sum())
    state_codes = Counter(value for value in state_values if len(value) == 2 and value.isalpha() and value in USPS)
    state_foreign = Counter(value for value in state_values if value and not (len(value) == 2 and value.isalpha() and value in USPS))
    bad_included_state = {value: int(count) for value, count in state_foreign.items() if not _state_text_like(value)}
    _check(
        checks,
        'state values for included rows',
        {
            'missing': missing_state,
            'usps_code_rows': int(sum(state_codes.values())),
            'non_usps_two_letter_text': {value: int(count) for value, count in state_foreign.items() if len(value) == 2 and value.isalpha()},
            'non_us_location_text': {value: int(count) for value, count in list(state_foreign.items())[:10]},
            'malformed_values': bad_included_state,
        },
        'state is a canonical USPS state or territory code, blank, or preserved non-US location text; full names were converted to codes and nothing is imputed',
        not bad_included_state,
        detail='Selected source member ' + SOURCE_NAME + '. Observed state values: ' + repr(dict(state_foreign.most_common(20))),
    )

    ynu_observed = {}
    ynu_bad = {}
    for column in ('hubzone_owned', 'women_owned', 'socially_economically_disadvantaged'):
        counts = Counter(frame[column].fillna('').astype(str))
        ynu_observed[column] = dict(counts)
        bad = {name: count for name, count in counts.items() if name not in ('Y', 'N', 'U', '')}
        if bad:
            ynu_bad[column] = bad
    _check(
        checks,
        'yes no unavailable values for included rows',
        ynu_observed,
        'each coded ownership field is Y, N, U, or blank, with unrecognized text never converted to Y or N',
        not ynu_bad,
        detail='Selected source member ' + SOURCE_NAME + '. Unrecognized coded values: ' + repr(ynu_bad),
    )

    emp = frame['number_employees'].dropna().astype(float)
    emp_non_integer = int(emp.map(lambda value: not float(value).is_integer()).sum())
    emp_negative = int((emp < 0).sum())
    emp_over = int((emp > 500).sum())
    _check(
        checks,
        'number_employees units and range for included rows',
        {
            'nonnull': int(len(emp)),
            'missing': int(frame['number_employees'].isna().sum()),
            'min': float(emp.min()) if len(emp) else None,
            'max': float(emp.max()) if len(emp) else None,
            'negative': emp_negative,
            'non_integer': emp_non_integer,
            'above_sbir_500_employee_limit': emp_over,
        },
        'whole nonnegative employee count in persons as published; blank stays blank and is never zero; the SBIR eligibility ceiling is 500 employees and values above it are reported as source evidence, not clipped',
        emp_negative == 0 and emp_non_integer == 0,
        detail='Selected source member ' + SOURCE_NAME + '. Observed employee range and counts are reported in the check object.',
        kind='unit',
    )

    amount = frame['award_amount'].dropna().astype(float)
    amount_negative = int((amount < 0).sum())
    _check(
        checks,
        'award_amount units for included rows',
        {
            'nonnull': int(len(amount)),
            'missing': int(frame['award_amount'].isna().sum()),
            'min': float(amount.min()) if len(amount) else None,
            'max': float(amount.max()) if len(amount) else None,
            'negative': amount_negative,
        },
        'nonnegative US dollars as published for the Phase I award only; blank stays blank and is never zero',
        amount_negative == 0,
        detail='Selected source member ' + SOURCE_NAME + '. Observed award amount range and counts are reported in the check object.',
        kind='unit',
    )

    target_values = Counter(int(value) for value in frame[TARGET])
    _check(
        checks,
        'target is binary and complete',
        dict(target_values),
        'target values are 0 and 1 with no missing row',
        set(target_values).issubset({0, 1}) and sum(target_values.values()) == len(frame),
        detail='Selected source member ' + SOURCE_NAME + '. Observed target counts: ' + repr(dict(target_values)),
    )

    id_unique = int(frame[ID_COL].nunique())
    _check(
        checks,
        'stable identifiers are unique',
        {'rows': int(len(frame)), 'unique_ids': id_unique},
        'one unique record_id per firm',
        id_unique == len(frame),
        detail='Selected source member ' + SOURCE_NAME + '. Duplicate rows: ' + str(int(len(frame) - id_unique)),
    )

    bad_fiscal = 0
    for _, row in frame.iterrows():
        dt = pd.to_datetime(row[TIME_COL], utc=True).tz_convert(None)
        y = int(row['award_year'])
        start = pd.Timestamp(year=y - 1, month=10, day=1)
        end = pd.Timestamp(year=y, month=10, day=1)
        if not (start <= dt < end):
            bad_fiscal += 1
    _check(
        checks,
        'award_date within award fiscal year',
        {'bad_rows': int(bad_fiscal), 'rows': int(len(frame))},
        'every shipped award_date lies in the half-open federal fiscal year of its award_year',
        bad_fiscal == 0,
        detail='Selected source member ' + SOURCE_NAME + '. Shipped award_date values must stay inside the award_year federal fiscal year.',
    )

    frame['_split'] = frame['award_year'].map(_split_for_year)
    if frame['_split'].isna().any():
        _fail(
            'Rows outside the fixed split years. Observed years '
            + str(sorted(frame['award_year'].unique().tolist())) + '.'
        )

    windows = {
        'train': ('2004-10-01T00:00:00.000', '2014-10-01T00:00:00.000'),
        'eval': ('2014-10-01T00:00:00.000', '2016-10-01T00:00:00.000'),
        'holdout': ('2016-10-01T00:00:00.000', '2018-10-01T00:00:00.000'),
    }
    split_ranges = {}
    split_ok = True
    for split in ('train', 'eval', 'holdout'):
        sub = frame[frame['_split'] == split]
        if len(sub):
            min_dt = pd.to_datetime(sub[TIME_COL]).min()
            max_dt = pd.to_datetime(sub[TIME_COL]).max()
        else:
            min_dt = pd.NaT
            max_dt = pd.NaT
        split_ranges[split] = {
            'rows': int(len(sub)),
            'min_date': min_dt.strftime('%Y-%m-%dT%H:%M:%S.000') if pd.notna(min_dt) else None,
            'max_date': max_dt.strftime('%Y-%m-%dT%H:%M:%S.000') if pd.notna(max_dt) else None,
            'positives': int(sub[TARGET].sum()) if len(sub) else 0,
        }
        start = pd.Timestamp(windows[split][0])
        end = pd.Timestamp(windows[split][1])
        if len(sub) == 0 or pd.isna(min_dt) or pd.isna(max_dt) or min_dt < start or max_dt >= end:
            split_ok = False
    _check(
        checks,
        'fixed chronological split ranges',
        split_ranges,
        'train FY2005-2014, eval FY2015-2016, holdout FY2017-2018, all nonempty and inside half-open federal fiscal windows',
        split_ok,
        detail='Selected source member ' + SOURCE_NAME + '. Observed split ranges: ' + repr(split_ranges),
    )

    positive_ok = all(split_ranges[s]['positives'] > 0 for s in windows)
    _check(
        checks,
        'positive events in every split',
        {s: split_ranges[s]['positives'] for s in windows},
        'at least one positive in train, eval and holdout',
        positive_ok,
        detail='Selected source member ' + SOURCE_NAME + '. Observed positive counts: ' + repr({s: split_ranges[s]['positives'] for s in windows}),
    )

    pos_bad = 0
    neg_bad = 0
    unknown_negative = 0
    for row in rows:
        key = row['_firm_key']
        agency = row['agency']
        year = row['award_year']
        phase2_years = phase2.get((key, agency), set())
        matched = any(future in phase2_years for future in range(year + 1, year + 5))
        if row[TARGET] == 1 and not matched:
            pos_bad += 1
        if row[TARGET] == 0 and matched:
            neg_bad += 1
        if row[TARGET] == 0:
            if any(future in unknown_phase.get((key, agency), set()) for future in range(year + 1, year + 5)):
                unknown_negative += 1
    _check(
        checks,
        'label matches later Phase II records',
        {
            'positive_without_match': pos_bad,
            'negative_with_match': neg_bad,
            'positives': int(frame[TARGET].sum()),
            'negatives': int(len(frame) - frame[TARGET].sum()),
        },
        'positive only when the same firm key and agency has a Phase II record in award_year plus one through plus four',
        pos_bad == 0 and neg_bad == 0,
        detail='Selected source member ' + SOURCE_NAME + '. Observed label disagreements: ' + repr({'positive_without_match': pos_bad, 'negative_with_match': neg_bad}),
    )

    _check(
        checks,
        'unknown phase never becomes a negative label',
        {
            'unknown_phase_records': int(phase_counts.get('unknown', 0)),
            'missing_phase_records': int(phase_counts.get('missing', 0)),
            'negative_rows_with_unknown_phase_in_window': unknown_negative,
            'excluded_unknown_phase_in_window': int(excluded.get('unknown_phase_in_window', 0)),
        },
        'no included negative row has an unknown or missing phase record inside its label window',
        unknown_negative == 0,
        detail='Selected source member ' + SOURCE_NAME + '. Observed unknown-phase exclusions and negative rows: ' + repr({'negative_rows_with_unknown_phase_in_window': unknown_negative, 'excluded_unknown_phase_in_window': int(excluded.get('unknown_phase_in_window', 0))}),
    )

    train_ids = set(frame.loc[frame['_split'] == 'train', ID_COL])
    eval_ids = set(frame.loc[frame['_split'] == 'eval', ID_COL])
    holdout_ids = set(frame.loc[frame['_split'] == 'holdout', ID_COL])
    overlaps = len(train_ids & eval_ids) + len(train_ids & holdout_ids) + len(eval_ids & holdout_ids)
    _check(
        checks,
        'split identifier disjointness',
        {'overlap': overlaps},
        'no record_id appears in more than one split',
        overlaps == 0,
        detail='Selected source member ' + SOURCE_NAME + '. Observed cross-split identifier overlap: ' + str(overlaps),
    )

    frame['number_employees'] = frame['number_employees'].astype('Int64')

    for split, filename in (
        ('train', output / 'public' / 'train.csv'),
        ('eval', output / 'public' / 'eval.csv'),
        ('holdout', output / 'private' / 'holdout.csv'),
    ):
        sub = frame[frame['_split'] == split][CSV_COLUMNS].sort_values([TIME_COL, ID_COL], kind='mergesort')
        sub.to_csv(filename, index=False)

    return checks
