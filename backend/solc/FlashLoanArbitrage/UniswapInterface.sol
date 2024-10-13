// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import '@uniswap/v3-periphery/contracts/interfaces/ISwapRouter.sol';
import '@uniswap/v3-periphery/contracts/libraries/TransferHelper.sol';

// Interface for Uniswap *NOT NECESSARY AS IMPORTED INTERFACE WILL BE USED
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
    ISwapRouter public immutable swapRouter;
    address public constant wbtcAddress;
    address public constant wethAddress;

    uint24 public constant poolFee = 3000;

    constructor(ISwapRouter _swapRouter) {
        wbtcAddress = _wbtcAddress;
        wethAddress = _wethAddress;
    }
    function swapWbtcForEth(uint256 wbtcAmountIn) external returns (uint256 ethAmountOut) {
        // Transfer the specified amount of DAI to this contract.
        TransferHelper.safeTransferFrom(DAI, msg.sender, address(this), amountIn);

        // Approve the router to spend DAI.
        TransferHelper.safeApprove(DAI, address(swapRouter), amountIn);
        
        ISwapRouter.ExactInputSingleParams memory params =
            ISwapRouter.ExactInputSingleParams({
                tokenIn: DAI,
                tokenOut: WETH9,
                fee: poolFee,
                recipient: msg.sender,
                deadline: block.timestamp,
                amountIn: wbtcAmountIn,
                amountOutMinimum: 0,
                sqrtPriceLimitX96: 0
                });

        // The call to `exactInputSingle` executes the swap.
        ethAmountOut = swapRouter.exactInputSingle(params);
}
}

