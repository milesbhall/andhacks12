"""
solana_proof.py
===============
Public, tamper-proof receipts for every signal.

When the pipeline flags a surprise, we write a short memo to Solana with the
SHA-256 hash of the full signal (speaker, statement, stance, z-score, the
markets we picked and which side). The block time proves WHEN we made the
call, and the hash proves WHAT we called, so nobody (including us) can
backfill a track record after the market moves.

Runs on Solana DEVNET by default: free, no real money. Uses the Memo program.

Keypair: solana_keypair.json (gitignored). Created on first run and funded
with a free devnet airdrop. If the airdrop faucet is rate limited, paste the
printed address into https://faucet.solana.com (devnet).

------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------
  pip install solders requests
  python solana_proof.py --address             # show / create the wallet
  python solana_proof.py --airdrop             # free devnet SOL
  python solana_proof.py --memo "hello"        # test memo
  python solana_proof.py --verify <signature>  # read a receipt back
------------------------------------------------------------------------
"""

import argparse
import base64
import hashlib
import json
import os
import time

import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KEYPAIR_PATH = os.path.join(SCRIPT_DIR, "solana_keypair.json")
SOLANA_CLUSTER = os.environ.get("SOLANA_CLUSTER", "devnet")
RPC_URLS = {
    "devnet": "https://api.devnet.solana.com",
    "mainnet-beta": "https://api.mainnet-beta.solana.com",
}
MEMO_PROGRAM_ID = "Memo1UhkJRfHyvLMcVucJwxXeuD728EqVDDwQDxFMNo"   # SPL Memo (deployed on devnet and mainnet)


def _rpc(method: str, params: list):
    resp = requests.post(RPC_URLS[SOLANA_CLUSTER], json={
        "jsonrpc": "2.0", "id": 1, "method": method, "params": params}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"Solana {method}: {data['error'].get('message', data['error'])}")
    return data["result"]


def load_keypair():
    from solders.keypair import Keypair

    if os.path.isfile(KEYPAIR_PATH):
        with open(KEYPAIR_PATH, encoding="utf-8") as f:
            return Keypair.from_bytes(bytes(json.load(f)))
    kp = Keypair()
    with open(KEYPAIR_PATH, "w", encoding="utf-8") as f:
        json.dump(list(bytes(kp)), f)
    print(f"Created new {SOLANA_CLUSTER} wallet {kp.pubkey()} ({os.path.basename(KEYPAIR_PATH)})")
    return kp


def balance_sol() -> float:
    return _rpc("getBalance", [str(load_keypair().pubkey())])["value"] / 1e9


def airdrop(sol: float = 1.0) -> str:
    if SOLANA_CLUSTER == "mainnet-beta":
        raise RuntimeError("No airdrops on mainnet.")
    return _rpc("requestAirdrop", [str(load_keypair().pubkey()), int(sol * 1e9)])


def send_memo(text: str) -> str:
    """Writes `text` on-chain with the Memo program. Returns the tx signature."""
    from solders.hash import Hash
    from solders.instruction import AccountMeta, Instruction
    from solders.message import Message
    from solders.pubkey import Pubkey
    from solders.transaction import Transaction

    kp = load_keypair()
    ix = Instruction(Pubkey.from_string(MEMO_PROGRAM_ID), text.encode("utf-8"),
                     [AccountMeta(kp.pubkey(), is_signer=True, is_writable=True)])
    blockhash = Hash.from_string(_rpc("getLatestBlockhash", [{"commitment": "finalized"}])["value"]["blockhash"])
    tx = Transaction([kp], Message([ix], kp.pubkey()), blockhash)
    return _rpc("sendTransaction", [base64.b64encode(bytes(tx)).decode(), {"encoding": "base64"}])


def signal_hash(signal: dict) -> str:
    canonical = json.dumps(signal, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def prove_signal(signal: dict) -> dict:
    """Hash the signal, put hash + a readable summary on-chain.

    Memo looks like:  andhacks|kevin_warsh|HAWKISH|z=+3.0|<sha256>
    """
    digest = signal_hash(signal)
    memo = (f"andhacks|{signal.get('speaker', '')}|{signal.get('direction', '')}|"
            f"z={float(signal.get('z', 0)):+.1f}|{digest}")
    sig = send_memo(memo)
    return {"cluster": SOLANA_CLUSTER, "signature": sig, "sha256": digest, "memo": memo,
            "explorer": explorer_url(sig)}


def explorer_url(signature: str) -> str:
    suffix = "" if SOLANA_CLUSTER == "mainnet-beta" else f"?cluster={SOLANA_CLUSTER}"
    return f"https://explorer.solana.com/tx/{signature}{suffix}"


def verify(signature: str, retries: int = 10) -> dict:
    """Reads a receipt back: block time + memo text."""
    for _ in range(retries):
        tx = _rpc("getTransaction", [signature, {"encoding": "jsonParsed", "commitment": "confirmed",
                                                 "maxSupportedTransactionVersion": 0}])
        if tx:
            memo = next((ix.get("parsed") for ix in tx["transaction"]["message"]["instructions"]
                         if ix.get("programId") == MEMO_PROGRAM_ID), None)
            return {"signature": signature, "slot": tx["slot"], "block_time": tx.get("blockTime"),
                    "memo": memo, "explorer": explorer_url(signature)}
        time.sleep(2)
    raise RuntimeError("Transaction not found yet.")


def main():
    parser = argparse.ArgumentParser(description=f"Signal receipts on Solana ({SOLANA_CLUSTER})")
    parser.add_argument("--address", action="store_true")
    parser.add_argument("--airdrop", action="store_true")
    parser.add_argument("--memo")
    parser.add_argument("--verify")
    args = parser.parse_args()

    if args.address:
        print(load_keypair().pubkey(), f"balance {balance_sol():.4f} SOL")
    elif args.airdrop:
        print("airdrop tx:", airdrop())
    elif args.memo:
        sig = send_memo(args.memo)
        print(sig, explorer_url(sig))
    elif args.verify:
        print(json.dumps(verify(args.verify), indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
