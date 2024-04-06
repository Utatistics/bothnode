pragma solidity ^0.8.0;

contract BalanceChecker {
    function getBalance(address account) public view returns (uint256) {
        return account.balance;
    }
}

