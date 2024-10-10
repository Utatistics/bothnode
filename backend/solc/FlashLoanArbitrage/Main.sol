// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "./CompoundInterface.sol"; 
import "./bZxInterface.sol";
import "./UniswapInterface.sol"; // Import Uniswap interface 
import "./FlashLoanProvider.sol"; // Aave or dYdX interface

/** define interface for Aave Lending Pool**/
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

/** inheriting from Aave's flashlaon exec logic **/
contract FlashLoanArbitrage is FlashLoanReceiverBase { 
    using SafeMath for uint256;

    // state variables
    ILendingPool public POOL; 
    UniswapInteraction public uniswapInteraction;
    constructor(address _addressProvider, address _uniswapAddress, address _WBTC, address _WETH) {
        POOL = ILendingPool(_addressProvider);  // Initialize Aave Lending Pool 
        uniswapInteraction = new UniswapInteraction(_uniswapAddress, _WBTC, _WETH);
    }

    /** main (i.e. body) function required, which will be called by AAVE once the flashloan is granted **/
    function executeOperation( // function signature 
        address[] calldata assets,
        uint256[] calldata amounts,
        uint256[] calldata premiums,
        address initiator,
        bytes calldata params
    ) external override returns (bool) 
    { // function body
        uint256 loanAmount = amounts[0];
        uint256 repaymentAmount = loanAmount.add(premiums[0]);

        // Step 2: Supply ETH as collateral and borrow WBTC
        _supplyCollateralToCompound(5500 ether);

        // Step 3: Open short position on bZx
        _openShortPosition(1300 ether);

        // Step 4: Swap WBTC for ETH via Uniswap *defined below
        uniswapInteraction._swapWbtcForEth(loanAmount);  // Pass in the loan amount or WBTC amount to swap

        // Step 5: Repay flash loan
        _repayFlashLoan(repaymentAmount);

        return true; // return true upon successful completion of flashloan
    }

    /** endpoint of the smart contract: will be called externally*/
    function executeFlashLoan(uint256 amount) external {
        address;
        assets[0] = address(WETH);  // WETH for ETH loans

        uint256;
        amounts[0] = amount;

        uint256;
        modes[0] = 0;  // Flash loan mode (0 = no debt)

        bytes memory params = "";

        // Initiate the flash loan from Aave’s POOL
        POOL.flashLoan(address(this), assets, amounts, modes, address(this), params, 0);
    }
}
