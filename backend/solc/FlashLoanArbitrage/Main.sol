// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@aave/core-v3/contracts/flashloan/base/FlashLoanReceiverBase.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "./PriceFeed.sol";
import "./UniswapInterface.sol";
import "./CompoundInterface.sol";
import "./bZxInterface.sol";

contract FlashLoanProvider is FlashLoanReceiverBase {
    CompoundInteraction public compound;
    bZxInteraction public bzx;
    UniswapInteraction public uniswap;

    constructor(
        // Aave (i.e flashloan provider)
        address _poolAddressesProvider,
        
        // Compund (i.e. lending and burrowing)
        address payable _cEtherAddress,
        address _cWBTCAddress,
        address _comptrollerAddress,
        uint256 _ethAmountAsCollateral,

        // bZx (i.e. leveraged short position) 
        address _bzxAddress,
        
        // Uniswap (i.e. currency swap)
        address _uniswapAddress,
        address _cETHAddres,
        address _cWBTCAddress,
        address _WBTC,
        address _WETH

    )
        FlashLoanReceiverBase(IPoolAddressesProvider(_poolAddressesProvider))
    {
        compound = CompoundInteraction(_cEtherAddress, _wbtcAddress, _comptrollerAddress, _underlyingAddress, _underlyingToSupplyAsCollateral);
        priceFeed = new Chainlink(_ethPriceFeedAddress, _wbtcPriceFeedAddress);
        uniswap = new UniswapInteraction(_uniswapAddress, _WBTC, _WETH);
        bzx = new bZxInteraction(_bzxAddress);
    }

    // Function to execute flash loan
    function executeFlashLoan(
        address[] calldata assets,
        uint256[] calldata amounts,
        uint256[] calldata modes,
        address onBehalfOf,
        bytes calldata params
    ) external {
        // Initiate the flash loan
        POOL.flashLoan(address(this), assets, amounts, modes, onBehalfOf, params, 0);
    }

    // Internal function to repay flash loan
    function _repayFlashLoan(address asset, uint256 amount) internal {
        // Approve the lending pool to pull the repayment
        IERC20(asset).approve(address(POOL), amount);

        // POOL will automatically pull the repayment from the contract once approved
    }

    // This function will be called by Aave once the flash loan is granted
    function executeOperation(
        address[] calldata assets,
        uint256[] calldata amounts,
        uint256[] calldata premiums,
        address initiator,
        bytes calldata params
    ) external override returns (bool) {
        uint256 loanAmount = amounts[0];
        uint256 repaymentAmount = loanAmount + premiums[0];  // Loan amount + fees

        // Step 2: Supply ETH as collateral and borrow WBTC
        compound.supplyETHAndBorrowWBTC(5500 ether);

        // Step 3: Open short position on bZx
        bzx.openShortPosition(1300 ether);

        // Step 4: Swap WBTC for ETH via Uniswap
        uniswap.swapWbtcForEth(loanAmount);  // Pass in the loan amount or WBTC amount to swap

        // Step 5: Repay flash loan
        _repayFlashLoan(assets[0], repaymentAmount);

        return true;
    }

}
