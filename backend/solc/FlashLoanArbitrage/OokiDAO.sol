// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface ERC20 {
    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function transfer(address recipient, uint256 amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
    function transferFrom(
        address sender,
        address recipient,
        uint256 amount
    ) external returns (bool);
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
}

interface IBZX{
    function underlyingToLoanPool(address) external view returns (address);
}

interface ILoanTokenLogicStandard {
    struct LoanOpenData {
        bytes32 loanId; // Unique loan identifier, 0 if new loan
        uint256 collateralAmount; // Amount of collateral provided
        uint256 loanAmount; // Total loan amount
        address collateralToken; // Address of collateral token
        address borrower; // Borrower's address
        uint256 interestRate; // Interest rate for the loan
        uint256 loanDuration;  // Duration of the loan
    }
    function marginTrade(
        bytes32 loanId,  // 0 if new loan
        uint256 leverageAmount,
        uint256 loanTokenSent,
        uint256 collateralTokenSent,
        address collateralTokenAddress,
        address trader,
        bytes memory loanDataBytes) 
        external 
        payable 
        returns (LoanOpenData memory);
}

contract OokiDAOInteraction {
    address public wethAddress; // Address of WETH token
    address public wbtcAddress; // Address of WBTC token
    address public underlyingTokenAddress;

    IBZX public bzx;
    ILoanTokenLogicStandard public loanTokenLogic;

    constructor(address _bzxAddress, address _wethAddress, address _wbtcAddress) {
        wethAddress = _wethAddress;
        wbtcAddress = _wbtcAddress;

        bzx = IBZX(_bzxAddress);
        underlyingTokenAddress = bzx.underlyingToLoanPool(wbtcAddress); // return the address to the pool for the underlying asset.
        loanTokenLogic = ILoanTokenLogicStandard(underlyingTokenAddress);
    }

    function openShortPosition(uint256 ethAmount, uint256 leverage) external {
        require(leverage >= 1, "Leverage must be at least 1");
        require(ethAmount > 0, "ETH amount must be greater than 0");

        // Calculate amounts
        uint256 collateralTokenSent = ethAmount; // Amount of ETH sent as collateral
        uint256 loanTokenSent = (ethAmount * leverage) - ethAmount; // Amount borrowed (ETH for WBTC)

        // Convert ETH to WETH before sending
        ERC20(wethAddress).transferFrom(msg.sender, address(this), collateralTokenSent);
        
        // Call marginTrade
        loanTokenLogic.marginTrade{value: collateralTokenSent}(
            0, // New loan
            leverage,
            loanTokenSent,
            collateralTokenSent,
            wethAddress, // Collateral token address (WETH)
            msg.sender,  // Trader address
            ""            // Arbitrary order data
        );
    }

    // Fallback function to receive ETH
    receive() external payable {}
}