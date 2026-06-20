"""Security profiles of the benchmarked schemes (qualitative analysis).

These are security *facts*, not runtime measurements -- they complement the performance
benchmark so the comparison covers BOTH overhead and security, as the assignment requires.
Security levels follow NIST SP 800-57 (AES/RSA strength) and the Homomorphic Encryption
Security Standard (CKKS / RLWE). The single most important distinction:

    AES and RSA protect data *at rest* and *in transit* -- to compute on it you must
    decrypt first, exposing plaintext to the compute party.
    CKKS additionally protects data *in use*: the compute party never sees plaintext;
    only the key holder decrypts the final result.
"""
from __future__ import annotations

import config


def security_profiles() -> list[dict]:
    """Return one structured security profile per scheme."""
    coeff_bits = sum(config.CKKS_COEFF_MOD_BIT_SIZES)
    return [
        {
            "scheme": "AES-256-GCM",
            "type": "Symmetric (authenticated)",
            "key": "256-bit key",
            "security_level": "~256-bit classical (~128-bit vs Grover)",
            "protects_at_rest_transit": True,
            "protects_in_use": False,
            "can_compute_on_ciphertext": False,
            "integrity": "Yes (GCM authentication tag)",
            "quantum_resistant": "Partly (key size resists Grover)",
            "exposure_during_compute": "Full plaintext exposed to the compute party",
            "notes": "Fast authenticated confidentiality for storage and transport.",
        },
        {
            "scheme": "RSA-2048-OAEP",
            "type": "Asymmetric",
            "key": "2048-bit modulus",
            "security_level": "~112-bit (NIST SP 800-57)",
            "protects_at_rest_transit": True,
            "protects_in_use": False,
            "can_compute_on_ciphertext": False,
            "integrity": "No (encryption only; signatures are separate)",
            "quantum_resistant": "No (broken by Shor's algorithm)",
            "exposure_during_compute": "Full plaintext exposed; bulk data needs hybrid + decryption",
            "notes": "Used for key wrapping / small payloads, not bulk data or computation.",
        },
        {
            "scheme": "CKKS (TenSEAL)",
            "type": "Homomorphic (RLWE lattice)",
            "key": f"poly_modulus_degree={config.CKKS_POLY_MODULUS_DEGREE}, "
                   f"coeff modulus={coeff_bits} bits",
            "security_level": "~128-bit (HE Security Standard, at these parameters)",
            "protects_at_rest_transit": True,
            "protects_in_use": True,
            "can_compute_on_ciphertext": True,
            "integrity": "No by default (confidentiality only; integrity needs verifiable compute)",
            "quantum_resistant": "Believed yes (lattice / RLWE based)",
            "exposure_during_compute": "None -- compute party never sees plaintext",
            "notes": "Approximate scheme. Caveat: giving an adversary the decrypted approximate "
                     "results can leak the secret key (Li-Micciancio 2021); mitigate with noise flooding.",
        },
    ]


def key_takeaways() -> list[str]:
    """Short, defensible bullet points for the report / presentation."""
    return [
        "HE's unique security benefit is protecting data **in use**: computation happens on "
        "ciphertext, so the compute party (e.g. an untrusted cloud) never sees plaintext. "
        "AES and RSA cannot do this -- they require decryption before any computation.",
        "This benefit is not free: the benchmark shows HE adds large runtime and ~10x "
        "ciphertext-size overhead versus AES. The trade-off is performance for "
        "confidentiality during processing.",
        "CKKS is lattice-based (RLWE) and therefore a post-quantum candidate, whereas "
        "RSA-2048 is broken by quantum (Shor). AES-256 stays strong against Grover.",
        "Scope matters: AES-GCM also provides integrity/authenticity; CKKS provides "
        "confidentiality only and is approximate, so results carry (measured) error and "
        "integrity must be added separately if needed.",
    ]
