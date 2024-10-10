// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Interface for Compound
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

    // Change visibility to public so it can be called externally
    function supplyCollateralToCompound(uint256 ethAmount) external {
        compound.supply(address(this), ethAmount);
    }

    function borrowWBTC(uint256 wbtcAmount) external {
        compound.borrow(address(this), wbtcAmount);
    }

    function repayBorrowedWBTC(uint256 amount) external {
        compound.repayBorrow(amount);
    }
}
