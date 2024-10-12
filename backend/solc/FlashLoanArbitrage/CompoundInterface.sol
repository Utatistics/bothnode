// SPDX-License-Identifier: MIT

pragma solidity ^0.8.6;

// declaring interfaces

interface CEth {
    function mint() external payable;
    function borrow(uint256) external returns (uint256);
    function repayBorrow() external payable;
    function borrowBalanceCurrent(address) external returns (uint256);
}

interface CErc20 {
    function mint(uint256) external returns (uint256);
    function borrow(uint256) external returns (uint256);
    function borrowRatePerBlock() external view returns (uint256);
    function borrowBalanceCurrent(address) external returns (uint256);
    function repayBorrow(uint256) external returns (uint256);
}

interface Comptroller {
    function markets(address) external returns (bool, uint256);
    function enterMarkets(address[] calldata)
        external
        returns (uint256[] memory);
    function getAccountLiquidity(address)
        external
        view
        returns (uint256, uint256, uint256);
}

interface PriceFeed {
    function getUnderlyingPrice(address cToken) external view returns (uint);
}

contract CompundInteraction {
    event CompundLog(string, uint256);

    // State variables 
    address payable public cEtherAddress;
    address public cWBTCAddress;
    address public comptrollerAddress;
    uint256 public ethAmountAsCollateral; // value in wei (i.e. ether)
    
    constructor(
        address payable _cEtherAddress,
        address _cWBTCAddress,
        address _comptrollerAddress,
        uint256 _ethAmountAsCollateral
    )
    {
        cEtherAddress = _cEtherAddress;
        cWBTCAddress = _cWBTCAddress;
        comptrollerAddress = _comptrollerAddress;
        ethAmountAsCollateral = _ethAmountAsCollateral;
    }

    function supplyETHAndBorrowWBTC(
) public returns (uint) {
        // initialize contract interfaces
        CEth cEth = CEth(cEtherAddress);
        CErc20 cWbtcToken = CErc20(cWBTCAddress);
        Comptroller comptroller = Comptroller(comptrollerAddress);

        // supplying the collateral
        cEth.mint{value: ethAmountAsCollateral}(); 
        
        // enter the market
        address[] memory markets = new address[](2); // initialize an address array with size of 2
        markets[0] = cEtherAddress;
        markets[1] = cWBTCAddress;
        
        uint256[] memory errorCodeArray = comptroller.enterMarkets(markets);
        for (uint256 i = 0; i < errorCodeArray.length; i++) {
            if (errorCodeArray[i] != 0) {
                revert("Comptroller.getAccountLiquidity failed.");
                }
            }

        // Get my account's total liquidity value in Compound
        (uint256 errorCodeLiquidity, uint256 liquidity, uint256 shortfall) = comptroller.getAccountLiquidity(address(this)); // tuple assignments 
        if (errorCodeLiquidity != 0) {
            revert("Comptroller.getAccountLiquidity failed.");
        }
        require(shortfall == 0, "account underwater");
        require(liquidity > 0, "account has excess collateral");
        emit CompundLog("Maximum ETH Borrow (borrow far less!)", liquidity);

        // Get the collateral factor for our collateral
        (bool _isListed, uint collateralFactorMantissa) = comptroller.markets(cEtherAddress);
        emit CompundLog('Collateral Factor', collateralFactorMantissa);

        // Calculate a safe borrowing amount (e.g., 90% of available liquidity)
        uint256 safeBorrowAmount = liquidity * 90 / 100; // 90% of liquidity

        // Borrow the calculated safe amount of WBTC
        cWbtcToken.borrow(safeBorrowAmount);

        // Check the current borrow balance for this contract's address
        uint256 borrows = cEth.borrowBalanceCurrent(address(this));
        emit CompundLog("Current ETH borrow amount", borrows);

        return borrows;

    }
    // Need this to receive ETH when `borrowEthExample` executes
    receive() external payable {}
}
