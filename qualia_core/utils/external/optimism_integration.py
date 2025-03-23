from web3 import Web3
from eth_account import Account
from eth_account.signers.local import LocalAccount
from config import Config
import json
import os

# Optimism Goerli testnet RPC URL
OPTIMISM_RPC_URL = "https://goerli.optimism.io"

# Load the contract ABIs
current_dir = os.path.dirname(os.path.abspath(__file__))
nft_json_path = os.path.join(current_dir, '..', 'contracts', 'YaaModuleNFT.json')
dao_json_path = os.path.join(current_dir, '..', 'contracts', 'DAOGovernance.json')

with open(nft_json_path, 'r') as file:
    nft_contract_data = json.load(file)
    nft_contract_abi = nft_contract_data['abi']
    nft_contract_bytecode = nft_contract_data['bytecode']

with open(dao_json_path, 'r') as file:
    dao_contract_data = json.load(file)
    dao_contract_abi = dao_contract_data['abi']
    dao_contract_bytecode = dao_contract_data['bytecode']

# Initialize Web3 instance for Optimism
w3 = Web3(Web3.HTTPProvider(OPTIMISM_RPC_URL))

def deploy_nft_contract_optimism():
    account: LocalAccount = Account.from_key(Config.PRIVATE_KEY)
    
    nft_contract = w3.eth.contract(abi=nft_contract_abi, bytecode=nft_contract_bytecode)
    
    tx = nft_contract.constructor().build_transaction({
        'from': account.address,
        'nonce': w3.eth.get_transaction_count(account.address),
        'gas': 3000000,
        'gasPrice': w3.eth.gas_price,
    })
    
    signed_tx = account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    
    return tx_receipt.contractAddress

def deploy_dao_contract_optimism(nft_contract_address):
    account: LocalAccount = Account.from_key(Config.PRIVATE_KEY)
    
    dao_contract = w3.eth.contract(abi=dao_contract_abi, bytecode=dao_contract_bytecode)
    
    tx = dao_contract.constructor(nft_contract_address).build_transaction({
        'from': account.address,
        'nonce': w3.eth.get_transaction_count(account.address),
        'gas': 3000000,
        'gasPrice': w3.eth.gas_price,
    })
    
    signed_tx = account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    
    return tx_receipt.contractAddress

def set_dao_governance_optimism(nft_contract_address, dao_contract_address):
    account: LocalAccount = Account.from_key(Config.PRIVATE_KEY)
    
    nft_contract = w3.eth.contract(address=nft_contract_address, abi=nft_contract_abi)
    
    tx = nft_contract.functions.setDAOGovernance(dao_contract_address).build_transaction({
        'from': account.address,
        'nonce': w3.eth.get_transaction_count(account.address),
        'gas': 3000000,
        'gasPrice': w3.eth.gas_price,
    })
    
    signed_tx = account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
    return tx_hash.hex()

def mint_nft_optimism(token_uri):
    account: LocalAccount = Account.from_key(Config.PRIVATE_KEY)
    
    nft_contract = w3.eth.contract(address=Config.NFT_CONTRACT_ADDRESS, abi=nft_contract_abi)
    
    tx = nft_contract.functions.createModuleNFT(account.address, token_uri).build_transaction({
        'from': account.address,
        'nonce': w3.eth.get_transaction_count(account.address),
        'gas': 3000000,
        'gasPrice': w3.eth.gas_price,
    })
    
    signed_tx = account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
    return tx_hash.hex()

def create_proposal_optimism(description):
    account: LocalAccount = Account.from_key(Config.PRIVATE_KEY)
    
    nft_contract = w3.eth.contract(address=Config.NFT_CONTRACT_ADDRESS, abi=nft_contract_abi)
    
    tx = nft_contract.functions.createProposal(description).build_transaction({
        'from': account.address,
        'nonce': w3.eth.get_transaction_count(account.address),
        'gas': 3000000,
        'gasPrice': w3.eth.gas_price,
    })
    
    signed_tx = account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
    return tx_hash.hex()

def vote_on_proposal_optimism(proposal_id, support):
    account: LocalAccount = Account.from_key(Config.PRIVATE_KEY)
    
    nft_contract = w3.eth.contract(address=Config.NFT_CONTRACT_ADDRESS, abi=nft_contract_abi)
    
    tx = nft_contract.functions.vote(proposal_id, support).build_transaction({
        'from': account.address,
        'nonce': w3.eth.get_transaction_count(account.address),
        'gas': 3000000,
        'gasPrice': w3.eth.gas_price,
    })
    
    signed_tx = account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
    return tx_hash.hex()

def get_proposal_optimism(proposal_id):
    dao_contract = w3.eth.contract(address=Config.DAO_CONTRACT_ADDRESS, abi=dao_contract_abi)
    proposal = dao_contract.functions.getProposal(proposal_id).call()
    return proposal
