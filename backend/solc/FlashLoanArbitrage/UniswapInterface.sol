// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import '@uniswap/v3-periphery/contracts/interfaces/ISwapRouter.sol';
import '@uniswap/v3-periphery/contracts/libraries/TransferHelper.sol';

interface IWETH {
    function deposit() external payable;
    function withdraw(uint256 amount) external;
}

// Contract for interacting with Uniswap
contract UniswapInteraction {
    // state variables
    address public swapRouterAddress;
    address public WBTC;
    address public WETH;

    ISwapRouter public immutable swapRouter;

    // Pool fee for WBTC/WETH on Uniswap (0.3% fee tier)
    uint24 public constant poolFee = 3000;

    constructor(address _swapRouterAddress, address _wbtcAddress, address _wethAddress) {
        swapRouterAddress = _swapRouterAddress;
        WBTC = _wbtcAddress;
        WETH = _wethAddress;

        swapRouter = ISwapRouter(swapRouterAddress);
    }
    function swapWbtcForEth(uint256 amountIn) external returns (uint256 amountOut) {
        // msg.sender must approve this contract to transfer WBTC

        // Transfer the specified amount of WBTC to this contract.
        TransferHelper.safeTransferFrom(WBTC, msg.sender, address(this), amountIn);

        // Approve the router to spend WBTC.
        TransferHelper.safeApprove(WBTC, address(swapRouter), amountIn);

        // Parameters for the swap, specifying WBTC -> WETH
        ISwapRouter.ExactInputSingleParams memory params =
            ISwapRouter.ExactInputSingleParams({
                tokenIn: WBTC,
                tokenOut: WETH,
                fee: poolFee,
                recipient: address(this), // The contract will receive WETH
                deadline: block.timestamp,
                amountIn: amountIn,
                amountOutMinimum: 0, // In production, use a non-zero minimum for safety
                sqrtPriceLimitX96: 0 // No price limit, executing at market price
            });

        // Perform the swap: WBTC -> WETH
        uint256 wethReceived = swapRouter.exactInputSingle(params);

        // Unwrap WETH to get ETH (since WETH is just ETH wrapped in an ERC-20 token)
        amountOut = unwrapWETH(wethReceived);

        // Transfer ETH to the user
        payable(msg.sender).transfer(amountOut);
    }

    function unwrapWETH(uint256 wethAmount) internal returns (uint256 ethAmount) {
        IWETH(WETH).withdraw(wethAmount); // Call the withdraw function to convert WETH to ETH
        ethAmount = wethAmount; // 1:1 conversion rate between WETH and ETH
    }

    // Receive function to accept ETH when unwrapping WETH
    receive() external payable {}
}

