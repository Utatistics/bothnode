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
    address private WBTC;
    address private WETH;

    // Constructor to initialize Uniswap and token addresses
    constructor(address _uniswapAddress, address _WBTC, address _WETH) {
        uniswap = UniswapInterface(_uniswapAddress);  // Uniswap contract interface
        WBTC = _WBTC;  // Store WBTC address
        WETH = _WETH;  // Store WETH address
    }

    // External function to swap WBTC for ETH (WETH)
    function swapWbtcForEth(uint256 WBTCAmount) external {
        // Dynamically declare and initialize the path for the token swap
        address;
        path[0] = WBTC;  // WBTC as the input token
        path[1] = WETH;  // WETH (ETH) as the output token

        // Perform the token swap via Uniswap
        uniswap.swapExactTokensForTokens(
            WBTCAmount,      // Amount of WBTC to swap
            0,               // Accept any amount of WETH (can be adjusted based on strategy)
            path,            // The path of the swap (WBTC -> WETH)
            address(this),   // Address receiving the WETH
            block.timestamp  // Deadline for transaction
        );
    }
}
