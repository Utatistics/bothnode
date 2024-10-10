// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Interface for Uniswap
interface UniswapInterface {
    function swapExactTokensForTokens(
        uint amountIn,
        uint amountOutMin, 
        address[] calldata path, // an array of paths to tokens
        address to, // recipient address
        uint deadline // transaction will be reverted if not met
    ) external returns (uint[] memory amounts);
}

// Contract for interacting with Uniswap
contract UniswapInteraction {
    UniswapInterface public uniswap;
    
    // State variables for WBTC and WETH addresses
    address[] public path; // SHOULD THESE BE PUBLIC?
    address public WBTC;
    address public WETH;

    // Constructor to initialize Uniswap and token addresses
    constructor(address _uniswapAddress, address _WBTC, address _WETH) {
        uniswap = UniswapInterface(_uniswapAddress);
        WBTC = _WBTC;
        WETH = _WETH;
    }

    // Internal function to swap WBTC for ETH
    function _swapWbtcForEth(uint256 WBTCAmount) internal {
        // Declare and initialize the path variable for WBTC -> ETH
        path[0] = WBTC;  // Use the WBTC address stored in the state variable
        path[1] = WETH;  // Use the WETH address stored in the state variable

        // Perform the token swap via Uniswap
        uniswap.swapExactTokensForTokens(
            WBTCAmount,       // Amount of WBTC to swap
            0,                // Accept any amount of ETH (can be adjusted based on strategy)
            path,
            address(this),    // Address receiving ETH
            block.timestamp   // Deadline for transaction
        );
    }
}
