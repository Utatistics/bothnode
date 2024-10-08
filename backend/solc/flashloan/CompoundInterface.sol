// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface CompoundInterface {
    function supply(address asset, uint256 amount) external;
    function borrow(address asset, uint256 amount) external;
    function repayBorrow(uint256 amount) external;
}

contract CompoundInteraction {
    CompoundInterface compound;

    constructor(address _compoundAddress) {
        compound = CompoundInterface(_compoundAddress);
    }

    function _supplyCollateralToCompound(uint256 ethAmount) internal {
        compound.supply(address(this), ethAmount);
    }

    function _borrowWBTC(uint256 wbtcAmount) internal {
        compound.borrow(address(this), wbtcAmount);
    }

    function _repayBorrowedWBTC(uint256 amount) internal {
        compound.repayBorrow(amount);
    }
}
