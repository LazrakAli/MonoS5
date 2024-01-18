.data
str:	.asciiz	"abcba"
ok:	.asciiz	"oui"
ko:	.asciiz	"non"


.text

main:

# prologue
	addiu	$29, $29, -8 #2 variable locale+0 appel pile
	

	
# n=len(str)	
	lui	$8,0x1001
	ori	$4,$8,0
	jal 	len
	sw	$2,0($29)
	
# res=est_palin
	lui	$8,0x1001
# str
	ori	$4,$8,0
# n-1
	lw	$5,0($29)
	addiu	$5,$5,-1
	jal 	est_palin
	ori	$9,$2,0	
	lui	$8,0x1001
# if (res==1)
	blez	$9, else
	addiu	$8,$8,6
# printf(%s,ok)
	ori	$4,$8,0
	ori	$2,$0,4
	syscall
	j	endif
else:		
# printf(%s,ko)
	addiu	$8,$8,10
	ori	$4,$8,0
	ori	$2,$0,4
	syscall
endif:
#epilogue
	addiu	$29,$29,8

# exit 
	ori	$2,$0,10
	syscall
	
len:
	ori	$2,$0,3
	jr $31

est_palin:
	ori	$2,$0,0
	jr	$31
	