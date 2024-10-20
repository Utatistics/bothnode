// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@aave/core-v3/contracts/flashloan/base/FlashLoanReceiverBase.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol"; // Import Ownable for access control

import "./CompoundInterface.sol";
import "./OokiDAO.sol";
import "./UniswapInterface.sol";

contract FlashLoanArbitrage is FlashLoanReceiverBase {
    CompoundInteraction public compound;
    OokiDAOInteraction public ookidao;
    UniswapInteraction public uniswap;

    constructor(
        /* Smart contract address */
        // Aave (i.e flashloan provider)
        address _poolAddressesProvider,
        
        // Compund (i.e. lending and burrowing)
        address _comptrollerAddress,
        address payable _cEthAddress,
        address _cWBTCAddress,
 
        // OokiDAO (i.e. leveraged short position) 
        address _bzxAddress,
        
        // Uniswap (i.e. currency swap)
        address _swapRouterAddress,  // multiple versions 
        
        // ERC20 token address
        address _wbtcAddress,
        address _wethAddress // WETH9

    )
        FlashLoanReceiverBase(IPoolAddressesProvider(_poolAddressesProvider))
    {
        compound = new CompoundInteraction(_comptrollerAddress, _cEthAddress, _cWBTCAddress);
        ookidao = new OokiDAOInteraction(_bzxAddress, _cEthAddress, _wbtcAddress);
        uniswap = new UniswapInteraction(_swapRouterAddress, _wbtcAddress, _wethAddress);
    }

    // Emdpoint function to execute flash loan
    function executeFlashLoan(
        address[] calldata assets,
        uint256[] calldata flashLoanAmounts,
        uint256[] calldata modes,
        address onBehalfOf,
        bytes calldata params
    ) external {
        // Initiate the flash loan
        POOL.flashLoan(address(this), assets, flashLoanAmounts, modes, onBehalfOf, params, 0);
    }

    // Internal function to repay flash loan
    function _repayFlashLoan(address asset, uint256 repayAmount) internal {
        // Approve the lending pool to pull the repayment
        IERC20(asset).approve(address(POOL), repayAmount);

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
        uint256 loanAmount = amounts[0];  // The amount you borrow for the flashloan as a whole
        uint256 repayAmount = loanAmount + premiums[0];  // Loan amount + fees

        // Step 2: Supply ETH as collateral and borrow WBTC from Compound
        uint256 wbtcBurrowAmount = compound.supplyETHAndBorrowWBTC(5500 ether);  // uint256 ethAmountAsCollateral -> uint256 wbtcBurrowAmount

        // Step 3: Open short position on OokiDAO
        ookidao.openShortPosition(1300 ether, 5);  // uint256 ethAmShortAmount, uint256 leverage

        // Step 4: Swap WBTC for ETH via Uniswap
        uniswap.swapWbtcForEth(wbtcBurrowAmount);  // uint256 amountIn

        // Step 5: Repay flash loan
        _repayFlashLoan(assets[0], repayAmount);

        return true;
    }
}

