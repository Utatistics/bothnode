// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.0;

contract Sample {
    function getBalance(address account) public view returns (uint256) {
        return account.balance;
    }
}

