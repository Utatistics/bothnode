// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@aave/core-v3/contracts/flashloan/base/FlashLoanReceiverBase.sol";

contract FlashLoanProvider is FlashLoanReceiverBase {
    constructor(address _poolAddressesProvider) FlashLoanReceiverBase(_poolAddressesProvider) {}

    // Additional functions to interact with Aave or dYdX can be added here
}
