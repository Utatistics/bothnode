// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface bZxInterface {
    function openPosition(address loanToken, uint256 leverage, uint256 loanAmount) external;
}

contract bZxInteraction {
    bZxInterface bZx;

    constructor(address _bZxAddress) {
        bZx = bZxInterface(_bZxAddress);
    }

    function _openShortPosition(uint256 ethAmount) internal {
        bZx.openPosition(address(this), 5, ethAmount);  // 5x leverage
    }
}
