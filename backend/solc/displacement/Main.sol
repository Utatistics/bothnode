// SPDX-License-Identifier: UNLICENSED
pragma solidity >=0.7.0 <0.9.0;

contract Displacement { // HOLDER CONTRACT
    // immutable so they cannot be changed after constructor:
    address public immutable AGENT;
    address public immutable ATTACKER;

    constructor(address agent, address attacker) {
        AGENT = agent;
        ATTACKER = attacker;
    }
    // define custom modifier that can modify the behavior of function when applied
    modifier onlyAllowed {
        // Allow either the agent or attacker to withdraw:
        require(msg.sender == AGENT || msg.sender == ATTACKER, "Only agent and attacker can interact!");
        _;
    }    

    // Allow receipt of funds:
    function receiveFunds() payable public {}

    function getContractBalance() public view returns (uint256) {
        // 'this' keyword represents current contract instance
        return address(this).balance;
    }

    function withdraw() public onlyAllowed {
        uint curBal = address(this).balance;
        require(curBal > 0, "Nothing to withdraw!");
        

        payable(msg.sender).transfer(curBal);
    }

}
