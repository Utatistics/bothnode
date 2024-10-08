// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "./CompoundInterface.sol"; 
import "./bZxInterface.sol";
import "./UniswapInterface.sol";
import "./FlashLoanProvider.sol";  // Aave or dYdX interface

interface ILendingPool {
    function flashLoan(
        address receiverAddress,
        address[] calldata assets,
        uint256[] calldata amounts,
        uint256[] calldata modes,
        address onBehalfOf,
        bytes calldata params,
        uint16 referralCode
    ) external;
}

contract FlashLoanArbitrage is FlashLoanReceiverBase {
    using SafeMath for uint256; // attach a library's functions to a data type

    ILendingPool public POOL;

    constructor(address _addressProvider) FlashLoanReceiverBase(_addressProvider) {
        POOL = ILendingPool(_addressProvider);  // Initialize POOL to Aave Lending Pool
    }

    function executeOperation(
        address[] calldata assets,
        uint256[] calldata amounts,
        uint256[] calldata premiums,
        address initiator,
        bytes calldata params
    ) external override returns (bool) /** function signature **/  
    { 
        uint256 loanAmount = amounts[0];
        uint256 repaymentAmount = loanAmount.add(premiums[0]);

        // Step 2: Supply ETH as collateral and borrow WBTC
        _supplyCollateralToCompound(5500 ether);

        // Step 3: Open short position on bZx
        _openShortPosition(1300 ether);

        // Step 4: Swap WBTC for ETH
        _swapWbtcForEth();

        // Step 5: Repay flash loan
        _repayFlashLoan(repaymentAmount);

        return true; // AAVE requires this boolean to finalize the transaction.
    }

    function executeFlashLoan(uint256 amount) external {
        address;
        assets[0] = address(WETH);  // WETH for ETH loans
        
        uint256;
        amounts[0] = amount;

        uint256;
        modes[0] = 0; // Flash loan mode

        bytes memory params = "";

        // Initiate the flash loan from Aave’s POOL
        POOL.flashLoan(address(this), assets, amounts, modes, address(this), params, 0);
    }
}
